"""Mortara EL250 XML format parser."""

from __future__ import annotations

import base64
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xmltodict
from dateutil.parser import parse as dparse
from dateutil.relativedelta import relativedelta

from ecgdatakit.exceptions import CorruptedFileError
from ecgdatakit.parsing.helpers.xml import find_tag
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


def _convert_to_datetime(date_string: str | None) -> datetime | None:
    """Parse a date string in various Mortara formats."""
    if date_string is None or date_string.strip() == "" or date_string == "0/0/0":
        return None
    date_string = date_string.strip()
    if re.match(r"^\d{8}$", date_string):
        return datetime.strptime(date_string, "%Y%m%d")
    elif "/" in date_string:
        return datetime.strptime(date_string, "%m/%d/%Y")
    else:
        return dparse(date_string)


def _decode_lead(base64_str: str) -> np.ndarray:
    """Decode a Base64-encoded lead into a float64 array."""
    raw = base64.b64decode(base64_str)
    signal = np.frombuffer(raw, dtype="<i2")
    return signal.astype(np.float64)


_DEMOGRAPHIC_MAPPING: dict[str, str] = {
    "First:": "first_name",
    "FName:&,": "first_name",
    "Name:": "last_name",
    "Last:": "last_name",
    "ID&ID:": "id",
    "ID:": "id",
    "DOB:": "dob",
    "Age&yr": "age",
    "Age:": "age",
    "Sex": "sex",
    "Sex:": "sex",
    "Height:": "height",
    "Weight:": "weight",
    "Race:": "race",
}


class MortaraEL250Parser(Parser):
    """Parser for Mortara EL250 XML ECG files."""

    FORMAT_NAME = "Mortara EL250"
    FORMAT_DESCRIPTION = "Mortara EL250 XML ECG format"
    FILE_EXTENSIONS = [".xml"]

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        try:
            text = header.decode("utf-8", errors="ignore")
            return "<ECG" in text and ("MORTARA" in text.upper() or "<CHANNEL" in text)
        except Exception:
            return False

    def parse(self, file_path: Path) -> ECGRecord:
        with open(file_path, "rb") as f:
            doc = xmltodict.parse(f.read())

        if "ECG" not in doc:
            raise CorruptedFileError(f"Missing ECG root element in {file_path}")

        ecg_root = doc["ECG"]
        record = ECGRecord(source_format="mortara_el250")

        xml_leads = find_tag(doc, "CHANNEL")
        if xml_leads is None:
            xml_leads = []
        elif isinstance(xml_leads, dict):
            xml_leads = [xml_leads]

        sampling_freq = 0
        sampling_duration = 0
        leads: list[Lead] = []

        # Extract ADC scaling: UNITS_PER_MV = ADC counts per millivolt
        units_per_mv = 0
        tc_node = ecg_root.get("TYPICAL_CYCLE", {})
        upm_str = ""
        if isinstance(tc_node, dict):
            upm_str = tc_node.get("@UNITS_PER_MV", "")
        if not upm_str:
            upm_str = ecg_root.get("@UNITS_PER_MV", "")
        if upm_str:
            try:
                units_per_mv = int(upm_str)
            except (ValueError, TypeError):
                pass
        adc_resolution = 1.0 / units_per_mv if units_per_mv > 0 else 1.0
        adc_units = "mV" if units_per_mv > 0 else ""

        for xml_lead in xml_leads:
            sampling_duration = int(xml_lead["@DURATION"])
            sampling_freq = int(xml_lead["@SAMPLE_FREQ"])
            samples = _decode_lead(xml_lead["@DATA"])
            raw = adc_resolution != 1.0
            leads.append(Lead(
                label=xml_lead["@NAME"],
                samples=samples,
                sample_rate=sampling_freq,
                resolution=adc_resolution,
                resolution_unit=adc_units,
                units="" if raw else adc_units,
                is_raw=raw,
            ))

        record.leads = leads

        xml_rep_beats = find_tag(doc, "TYPICAL_CYCLE_CHANNEL")
        rep_beats: dict[str, list] = {}
        if xml_rep_beats is not None:
            if isinstance(xml_rep_beats, dict):
                xml_rep_beats = [xml_rep_beats]
            for rb in xml_rep_beats:
                rep_beats[rb["@NAME"]] = _decode_lead(rb["@DATA"]).tolist()
        record.raw_metadata["representative_beats"] = rep_beats

        source = ecg_root.get("SOURCE", {})
        record.raw_metadata["signal_characteristics"] = {
            "sampling_rate": sampling_freq,
            "lowpass": source.get("@LOW_PASS_FILTER", ""),
            "number_of_leads": len(leads),
            "device": source.get("@MODEL", ""),
            "device_manufacturer": source.get("@MANUFACTURER", ""),
            "device_serial_number": source.get("@ACQUIRING_DEVICE_SERIAL_NUMBER", ""),
            "acquisition_type": source.get("@TYPE", ""),
        }

        device = DeviceInfo(
            manufacturer=source.get("@MANUFACTURER", ""),
            model=source.get("@MODEL", ""),
            serial_number=source.get("@ACQUIRING_DEVICE_SERIAL_NUMBER", ""),
            software_version=source.get("@SOFTWARE_VERSION", ""),
            acquisition_type=source.get("@TYPE", ""),
        )

        filters = FilterSettings()
        lp = source.get("@LOW_PASS_FILTER")
        if lp is not None:
            try:
                filters.lowpass = float(lp)
            except (ValueError, TypeError):
                pass
        hp = source.get("@HIGH_PASS_FILTER")
        if hp is not None:
            try:
                filters.highpass = float(hp)
            except (ValueError, TypeError):
                pass
        nf = source.get("@NOTCH_FILTER")
        if nf is not None:
            try:
                filters.notch = float(nf)
                if filters.notch:
                    filters.notch_active = True
            except (ValueError, TypeError):
                pass

        patient = PatientInfo()
        demo_fields_node = ecg_root.get("DEMOGRAPHIC_FIELDS")
        if demo_fields_node:
            raw_fields = find_tag(demo_fields_node, "DEMOGRAPHIC_FIELD")
            if raw_fields is not None:
                if isinstance(raw_fields, dict):
                    raw_fields = [raw_fields]
                demo: dict[str, str] = {}
                for item in raw_fields:
                    label = item.get("@LABEL", "")
                    value = item.get("@VALUE", "")
                    if label in _DEMOGRAPHIC_MAPPING:
                        demo[_DEMOGRAPHIC_MAPPING[label]] = value.replace("'", "").replace('"', "")

                patient.first_name = demo.get("first_name", "")
                patient.last_name = demo.get("last_name", "")
                patient.patient_id = demo.get("id", "")
                patient.sex = demo.get("sex", "U")

                age_str = demo.get("age", "")
                if age_str.isdigit():
                    patient.age = int(age_str)

                dob_str = demo.get("dob", "")
                if dob_str:
                    patient.birth_date = _convert_to_datetime(dob_str)

                height_str = demo.get("height", "")
                if height_str:
                    try:
                        patient.height = float(height_str)
                    except ValueError:
                        pass
                weight_str = demo.get("weight", "")
                if weight_str:
                    try:
                        patient.weight = float(weight_str)
                    except ValueError:
                        pass
                race_str = demo.get("race", "")
                if race_str:
                    patient.race = race_str

        record.patient = patient

        recording = RecordingInfo()

        acq_time = ecg_root.get("@ACQUISITION_TIME")
        if acq_time:
            try:
                recording.date = datetime.strptime(acq_time, "%Y%m%d%H%M%S")
            except ValueError:
                pass

        if sampling_freq > 0 and sampling_duration > 0:
            duration_s = sampling_duration / sampling_freq
            recording.duration = timedelta(seconds=duration_s)
            if recording.date:
                recording.end_date = recording.date + recording.duration

        record.recording = recording
        record.recording.device = device
        record.recording.acquisition.filters = filters

        if patient.birth_date and recording.date:
            record.raw_metadata["age_at_ecg"] = relativedelta(
                recording.date, patient.birth_date
            ).years

        annotations: dict[str, str] = {}
        num_qrs = ecg_root.get("@NUM_QRS")
        if num_qrs:
            annotations["num_qrs"] = str(num_qrs)
        vent_rate = ecg_root.get("@VENT_RATE")
        if vent_rate:
            annotations["vent_rate"] = str(vent_rate)

        typical_cycle = ecg_root.get("TYPICAL_CYCLE", {})
        skip_keys = {"BITS", "FORMAT", "UNITS_PER_MV", "DURATION",
                     "SAMPLE_FREQ", "ENCODING", "TYPICAL_CYCLE_CHANNEL"}
        for k, v in typical_cycle.items():
            if k.startswith("@"):
                key_clean = k[1:].upper()
                if key_clean not in skip_keys:
                    annotations[key_clean.lower()] = str(v)

        record.annotations = annotations

        measurements = GlobalMeasurements()
        meas_sources = {}
        meas_sources.update(
            {k.lstrip("@"): v for k, v in typical_cycle.items() if k.startswith("@")}
        )
        for k, v in ecg_root.items():
            if isinstance(v, str) and not k.startswith("#"):
                meas_sources.setdefault(k.lstrip("@"), v)

        _meas_map = {
            "VENT_RATE": "heart_rate",
            "QRS_DURATION": "qrs_duration",
            "QT_INT": "qt_interval",
            "PR_INT": "pr_interval",
            "P_AXIS": "p_axis",
            "QRS_AXIS": "qrs_axis",
            "T_AXIS": "t_axis",
            "NUM_QRS": "qrs_count",
            "RR_INT": "rr_interval",
            "RR_INTERVAL": "rr_interval",
            "QTC_INT": "qtc_bazett",
            "QT_CORRECTED": "qtc_bazett",
            "QTC_INTERVAL": "qtc_bazett",
        }
        for xml_key, attr_name in _meas_map.items():
            raw = meas_sources.get(xml_key)
            if raw is not None:
                try:
                    setattr(measurements, attr_name, int(raw))
                except (ValueError, TypeError):
                    pass
        record.measurements = measurements

        rep_beat_rate = 0
        tc = ecg_root.get("TYPICAL_CYCLE", {})
        if tc:
            try:
                rep_beat_rate = int(tc.get("@SAMPLE_FREQ", 0))
            except (ValueError, TypeError):
                pass
        if not rep_beat_rate:
            rep_beat_rate = sampling_freq

        for lead_name, samples_list in rep_beats.items():
            raw = adc_resolution != 1.0
            record.median_beats.append(Lead(
                label=lead_name,
                samples=np.array(samples_list, dtype=np.float64),
                sample_rate=rep_beat_rate,
                resolution=adc_resolution,
                resolution_unit=adc_units,
                units="" if raw else adc_units,
                is_raw=raw,
            ))

        interp = Interpretation()
        interp.source = "machine"
        statements: list[tuple[str, str]] = []

        diag_node = find_tag(doc, "DIAGNOSIS")
        if diag_node is not None:
            if isinstance(diag_node, dict):
                diag_node = [diag_node]
            if isinstance(diag_node, list):
                for d in diag_node:
                    if isinstance(d, dict):
                        stmt = d.get("@TEXT") or d.get("@DESCRIPTION") or d.get("#text", "")
                        if stmt:
                            statements.append((str(stmt).strip(), ""))
                    elif isinstance(d, str) and d.strip():
                        statements.append((d.strip(), ""))

        interp_node = find_tag(doc, "INTERPRETATION")
        if interp_node is not None:
            if isinstance(interp_node, dict):
                interp_node = [interp_node]
            if isinstance(interp_node, list):
                for item in interp_node:
                    if isinstance(item, dict):
                        stmt = item.get("@TEXT") or item.get("@DESCRIPTION") or item.get("#text", "")
                        if stmt:
                            statements.append((str(stmt).strip(), ""))
                    elif isinstance(item, str) and item.strip():
                        statements.append((item.strip(), ""))

        diag_stmt_node = find_tag(doc, "DIAGNOSIS_STATEMENT")
        if diag_stmt_node is not None:
            if isinstance(diag_stmt_node, dict):
                diag_stmt_node = [diag_stmt_node]
            if isinstance(diag_stmt_node, list):
                for item in diag_stmt_node:
                    if isinstance(item, dict):
                        stmt = item.get("@TEXT") or item.get("@DESCRIPTION") or item.get("#text", "")
                        if stmt:
                            statements.append((str(stmt).strip(), ""))
                    elif isinstance(item, str) and item.strip():
                        statements.append((item.strip(), ""))

        severity_val = ecg_root.get("@SEVERITY") or ecg_root.get("@DIAGNOSIS_SEVERITY", "")
        if severity_val:
            interp.severity = str(severity_val).strip().upper()

        interp.statements = statements
        if statements:
            record.interpretation = interp

        record.recording.acquisition.signal = SignalCharacteristics(
            sample_rate=sampling_freq,
            bits_per_sample=16,
            signal_signed=True,
            number_channels_valid=len(record.leads),
            number_channels_allocated=len(xml_leads),
            data_encoding="base64_int16le",
            compression="none",
        )

        record.raw_metadata["filepath"] = str(file_path)

        return record

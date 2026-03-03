"""Philips Sierra ECG XML format parser."""

from __future__ import annotations

from base64 import b64decode
from datetime import datetime, timedelta
from math import floor
from pathlib import Path
from typing import cast
from xml.dom.minidom import Attr, Document

import numpy as np
import numpy.typing as npt
import xmltodict
from defusedxml import minidom

from ecgdatakit.parsing.codecs.xli import xli_decode
from ecgdatakit.exceptions import MissingElementError, UnsupportedFormatError
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

_SUPPORTED_TYPES = {"SierraECG", "PhilipsECG"}
_SUPPORTED_VERSIONS = {"1.03", "1.04", "1.04.01", "1.04.02"}



class XMLField:
    """Declarative descriptor for extracting a typed value from an xmltodict tree."""

    def __init__(self, root_node_name: str, field_path: str, dtype: str) -> None:
        supported = ("str", "int", "float", "bool")
        if dtype not in supported:
            raise ValueError(f"Invalid dtype, supported values are {supported}")
        self._root_node_name = root_node_name
        self._field_path = field_path
        self._dtype = dtype

    def get_value(self, doc: dict) -> str | int | float | bool | None:
        extracted = find_tag(doc, self._root_node_name)
        if extracted is None:
            return None
        data = read_path(extracted, self._field_path)
        if data is None or data == "":
            return None
        try:
            if self._dtype == "str":
                return str(data)
            elif self._dtype == "int":
                return int(data)
            elif self._dtype == "float":
                return float(data)
            elif self._dtype == "bool":
                return str(data).lower() in ("yes", "true", "t", "1")
        except (ValueError, TypeError):
            return None
        return None



PATIENT_FIELDS: dict[str, XMLField] = {
    "patient_id": XMLField("generalpatientdata", "patientid", "str"),
    "first_name": XMLField("generalpatientdata", "firstname", "str"),
    "last_name": XMLField("generalpatientdata", "lastname", "str"),
    "age": XMLField("generalpatientdata", "age/years", "int"),
    "sex": XMLField("generalpatientdata", "sex", "str"),
    "weight": XMLField("generalpatientdata", "kg", "float"),
    "height": XMLField("generalpatientdata", "cm", "float"),
    "race": XMLField("generalpatientdata", "race", "str"),
}

RECORDING_FIELDS: dict[str, XMLField] = {
    "start_date": XMLField("dataacquisition", "@date", "str"),
    "start_time": XMLField("dataacquisition", "@time", "str"),
    "duration": XMLField("parsedwaveforms", "@durationperchannel", "int"),
}

SIGNAL_CHARACTERISTICS_FIELDS: list[dict[str, XMLField | int]] = [
    {
        "origin": XMLField("dataacquisition", "machine/#text", "str"),
        "sampling_rate": XMLField("signalcharacteristics", "samplingrate", "int"),
        "resolution": XMLField("signalcharacteristics", "resolution", "int"),
        "hipass": XMLField("signalcharacteristics", "hipass", "float"),
        "lowpass": XMLField("signalcharacteristics", "lowpass", "float"),
        "acsetting": XMLField("signalcharacteristics", "acsetting", "int"),
        "notch_filtered": XMLField("signalcharacteristics", "notchfiltered", "bool"),
        "notch_filter_freqs": XMLField("signalcharacteristics", "notchfilterfreqs", "int"),
        "acquisition_type": XMLField("signalcharacteristics", "acquisitiontype", "str"),
        "bits_per_sample": XMLField("signalcharacteristics", "bitspersample", "int"),
        "signal_offset": XMLField("signalcharacteristics", "signaloffset", "int"),
        "signal_signed": XMLField("signalcharacteristics", "signalsigned", "bool"),
        "number_channels_allocated": XMLField("signalcharacteristics", "numberchannelsallocated", "int"),
        "number_channels_valid": XMLField("signalcharacteristics", "numberchannelsvalid", "int"),
        "electrode_placement": XMLField("signalcharacteristics", "electrodeplacement", "str"),
    },
    {
        "data_encoding": XMLField("parsedwaveforms", "@dataencoding", "str"),
        "compression": XMLField("parsedwaveforms", "@compression", "str"),
        "number_of_leads": XMLField("parsedwaveforms", "@numberofleads", "int"),
        "duration": XMLField("parsedwaveforms", "@durationperchannel", "int"),
        "sampling_rate": XMLField("parsedwaveforms", "@samplespersecond", "int"),
        "resolution": XMLField("parsedwaveforms", "@resolution", "int"),
        "signal_offset": XMLField("parsedwaveforms", "@signaloffset", "int"),
        "signal_signed": XMLField("parsedwaveforms", "@signalsigned", "bool"),
        "bits_per_sample": XMLField("parsedwaveforms", "@bitspersample", "int"),
        "hipass": XMLField("parsedwaveforms", "@hipass", "float"),
        "lowpass": XMLField("parsedwaveforms", "@lowpass", "float"),
        "notch_filtered": XMLField("parsedwaveforms", "@notchfiltered", "bool"),
        "notch_filter_freqs": XMLField("parsedwaveforms", "@notchfilterfreqs", "int"),
        "artifact_filtered": XMLField("parsedwaveforms", "@artfiltered", "bool"),
        "waveform_modified": XMLField("parsedwaveforms", "@waveformmodified", "bool"),
        "origin": XMLField("parsedwaveforms", "@modifiedby", "str"),
        "upsampled": XMLField("parsedwaveforms", "@upsampled", "bool"),
        "upsampling_method": XMLField("parsedwaveforms", "@upsamplemethod", "str"),
        "downsampled": XMLField("parsedwaveforms", "@downsampled", "bool"),
    },
]

ANNOTATION_FIELDS: dict[str, XMLField] = {
    "interpretation_date": XMLField("interpretation", "@date", "str"),
    "interpretation_time": XMLField("interpretation", "@time", "str"),
    "criteria_version": XMLField("interpretation", "@criteriaversion", "str"),
    "criteria_version_date": XMLField("interpretation", "@criteriaversiondate", "str"),
    "heartrate": XMLField("globalmeasurements", "heartrate/#text", "str"),
    "rrint": XMLField("globalmeasurements", "rrint/#text", "str"),
    "print": XMLField("globalmeasurements", "print/#text", "str"),
    "qonset": XMLField("globalmeasurements", "qonset/#text", "str"),
    "qrsdur": XMLField("globalmeasurements", "qrsdur/#text", "str"),
    "qtint": XMLField("globalmeasurements", "qtint/#text", "str"),
    "qtcb": XMLField("globalmeasurements", "qtcb/#text", "str"),
    "qtcf": XMLField("globalmeasurements", "qtcf", "str"),
    "pfrontaxis": XMLField("globalmeasurements", "pfrontaxis/#text", "str"),
    "i40frontaxis": XMLField("globalmeasurements", "i40frontaxis/#text", "str"),
    "t40frontaxis": XMLField("globalmeasurements", "t40frontaxis/#text", "str"),
    "qrsfrontaxis": XMLField("globalmeasurements", "qrsfrontaxis/#text", "str"),
    "stfrontaxis": XMLField("globalmeasurements", "stfrontaxis/#text", "str"),
    "tfrontaxis": XMLField("globalmeasurements", "tfrontaxis/#text", "str"),
    "phorizaxis": XMLField("globalmeasurements", "phorizaxis/#text", "str"),
    "i40horizaxis": XMLField("globalmeasurements", "i40horizaxis/#text", "str"),
    "t40horizaxis": XMLField("globalmeasurements", "t40horizaxis/#text", "str"),
    "qrshorizaxis": XMLField("globalmeasurements", "qrshorizaxis/#text", "str"),
    "sthorizaxis": XMLField("globalmeasurements", "sthorizaxis/#text", "str"),
    "thorizaxis": XMLField("globalmeasurements", "thorizaxis/#text", "str"),
    "md_signature_line": XMLField("interpretation", "mdsignatureline", "str"),
    "severity_code": XMLField("severity", "@code", "str"),
    "severity": XMLField("severity", "#text", "str"),
}



def _get_node(xdoc: Document, tag_name: str) -> Document:
    xelt = _get_opt_node(xdoc, tag_name)
    if xelt is None:
        raise MissingElementError(tag_name)
    return xelt


def _get_opt_node(xdoc: Document, tag_name: str) -> Document | None:
    for xelt in xdoc.getElementsByTagName(tag_name):
        return cast(Document, xelt)
    return None


def _get_nodes(xdoc: Document, tag_name: str) -> list[Document]:
    return [cast(Document, xelt) for xelt in xdoc.getElementsByTagName(tag_name)]


def _get_attr(xdoc: Document, attr_name: str, default: str | None = None) -> str:
    if attr_name in xdoc.attributes:
        return _get_text(xdoc.attributes[attr_name])
    if default is None:
        raise MissingElementError(attr_name)
    return default


def _get_text(xdoc: Document | Attr) -> str:
    rc: list[str] = []
    for node in xdoc.childNodes:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return "".join(rc)


def _assert_version(root: Document) -> tuple[str, str]:
    doc_info = _get_node(root, "documentinfo")
    doc_type = _get_text(_get_node(doc_info, "documenttype")).strip()
    doc_ver = _get_text(_get_node(doc_info, "documentversion")).strip()
    if doc_type not in _SUPPORTED_TYPES or doc_ver not in _SUPPORTED_VERSIONS:
        raise UnsupportedFormatError(f"Files of type {doc_type} {doc_ver} are unsupported")
    return doc_type, doc_ver


def _get_or_create_labels(signal_details: Document, parsed_waveforms: Document) -> list[str]:
    lead_labels = _get_attr(parsed_waveforms, "leadlabels", "")
    if lead_labels != "":
        lead_count = int(_get_attr(parsed_waveforms, "numberofleads"))
        return lead_labels.split(" ")[:lead_count]
    good_channels = int(_get_text(_get_node(signal_details, "numberchannelsallocated")))
    leads_used = _get_text(_get_node(signal_details, "acquisitiontype"))
    return [_get_lead_name(leads_used, x + 1) for x in range(good_channels)]


def _get_lead_name(leads_used: str, index: int) -> str:
    if leads_used in ("STD-12", "10-WIRE"):
        names = {1: "I", 2: "II", 3: "III", 4: "aVR", 5: "aVL", 6: "aVF"}
        if index in names:
            return names[index]
        if 6 < index <= 12:
            return f"V{index - 6}"
    return f"Channel {index}"


def _split_leads(waveform_data: bytes, lead_count: int, samples: int) -> list[npt.NDArray[np.int16]]:
    all_samples: npt.NDArray[np.int16] = np.frombuffer(waveform_data, dtype=np.int16)
    leads: list[npt.NDArray[np.int16]] = []
    offset = 0
    while offset < lead_count * samples:
        leads.append(all_samples[offset: offset + samples])
        offset += samples
    return leads


def _infer_compression(parsed_waveforms: Document) -> str:
    return _get_attr(
        parsed_waveforms,
        "compressmethod",
        _get_attr(parsed_waveforms, "compression", "Uncompressed"),
    )


def _get_waveform_data(
    signal_details: Document,
    parsed_waveforms: Document,
    labels: list[str],
) -> list[npt.NDArray[np.int16]]:
    sampling_freq = int(_get_text(_get_node(signal_details, "samplingrate")))
    duration = int(_get_attr(parsed_waveforms, "durationperchannel"))
    sample_count = int(duration * (sampling_freq / 1000))

    encoding = _get_attr(parsed_waveforms, "dataencoding")
    if encoding != "Base64":
        raise UnsupportedFormatError(f"Waveform data encoding unsupported: {encoding}")
    waveform_data = b64decode(_get_text(parsed_waveforms))

    compression = _infer_compression(parsed_waveforms)
    if compression != "Uncompressed":
        if compression == "XLI":
            return xli_decode(waveform_data, labels)
        raise UnsupportedFormatError(f"Compression unsupported: {compression}")

    return _split_leads(waveform_data, len(labels), sample_count)


def _read_sierra_leads(filepath: Path) -> tuple[list[str], list[npt.NDArray[np.int16]], int, int]:
    """Read leads from a Sierra ECG file using DOM parsing.

    Returns (labels, sample_arrays, sampling_freq, duration_ms).
    """
    xdom = minidom.parse(str(filepath))
    root = _get_node(xdom, "restingecgdata")
    _assert_version(root)

    signal_details = _get_node(_get_node(root, "dataacquisition"), "signalcharacteristics")
    parsed_waveforms = _get_node(root, "parsedwaveforms")

    sampling_freq = int(_get_text(_get_node(signal_details, "samplingrate")))
    duration = int(_get_attr(parsed_waveforms, "durationperchannel"))

    labels = _get_or_create_labels(signal_details, parsed_waveforms)
    waveform_data = _get_waveform_data(signal_details, parsed_waveforms, labels)

    if len(waveform_data) >= 6:
        lead_i = waveform_data[0]
        lead_ii = waveform_data[1]
        lead_iii = waveform_data[2]
        lead_avr = waveform_data[3]
        lead_avl = waveform_data[4]
        lead_avf = waveform_data[5]

        for i in range(len(lead_iii)):
            lead_iii[i] = lead_ii[i] - lead_i[i] - lead_iii[i]
        for i in range(len(lead_avr)):
            lead_avr[i] = -lead_avr[i] - floor((lead_i[i] + lead_ii[i]) / 2)
        for i in range(len(lead_avl)):
            lead_avl[i] = floor((lead_i[i] - lead_iii[i]) / 2) - lead_avl[i]
        for i in range(len(lead_avf)):
            lead_avf[i] = floor((lead_ii[i] + lead_iii[i]) / 2) - lead_avf[i]

    return labels, waveform_data, sampling_freq, duration



class SierraXMLParser(Parser):
    """Parser for Philips Sierra ECG XML files."""

    FORMAT_NAME = "Philips Sierra XML"
    FORMAT_DESCRIPTION = "Philips Sierra ECG XML format"
    FILE_EXTENSIONS = [".xml"]

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        try:
            text = header.decode("utf-8", errors="ignore").lower()
            return "<restingecgdata" in text
        except Exception:
            return False

    def parse(self, file_path: Path) -> ECGRecord:
        with open(file_path, "rb") as f:
            doc = xmltodict.parse(f.read())

        record = ECGRecord(source_format="sierra_xml")
        record.patient = self._read_patient(doc)
        record.recording = self._read_recording(doc)
        record.leads = self._read_leads(file_path)
        record.annotations = self._read_annotations(doc)
        record.device = self._read_device(doc)
        record.filters = self._read_filters(doc)
        record.measurements = self._read_measurements(doc)
        record.interpretation = self._read_interpretation(doc)
        record.signal = self._read_signal(doc)
        record.raw_metadata["signal_characteristics"] = self._read_signal_characteristics(doc)
        record.raw_metadata["filepath"] = str(file_path)

        lead_measurements = self._read_lead_measurements(doc)
        if lead_measurements is not None:
            record.raw_metadata["lead_measurements"] = lead_measurements

        order_info = self._read_order_info(doc)
        if order_info is not None:
            record.raw_metadata["order_info"] = order_info
            if isinstance(order_info, dict):
                if not record.recording.referring_physician:
                    for key in ("referringphysician", "orderingphysician"):
                        val = order_info.get(key)
                        if val and isinstance(val, str):
                            record.recording.referring_physician = val
                            break
                if not record.recording.room:
                    val = order_info.get("room")
                    if val and isinstance(val, str):
                        record.recording.room = val

        # Extract technician from dataacquisition node
        tech_field = XMLField("dataacquisition", "operatorid", "str")
        tech = tech_field.get_value(doc)
        if tech and not record.recording.technician:
            record.recording.technician = str(tech)

        if record.leads:
            record.recording.sample_rate = record.leads[0].sample_rate

        return record

    def _read_patient(self, doc: dict) -> PatientInfo:
        info = PatientInfo()
        for field_name, xml_field in PATIENT_FIELDS.items():
            value = xml_field.get_value(doc)
            if value is not None:
                if field_name == "patient_id":
                    info.patient_id = str(value)
                elif field_name == "first_name":
                    info.first_name = str(value)
                elif field_name == "last_name":
                    info.last_name = str(value)
                elif field_name == "age":
                    info.age = int(value)
                elif field_name == "sex":
                    info.sex = str(value)
                elif field_name == "weight":
                    info.weight = float(value)
                elif field_name == "height":
                    info.height = float(value)
                elif field_name == "race":
                    info.race = str(value)
        return info

    def _read_recording(self, doc: dict) -> RecordingInfo:
        info = RecordingInfo()
        start_date = RECORDING_FIELDS["start_date"].get_value(doc)
        start_time = RECORDING_FIELDS["start_time"].get_value(doc)
        duration_ms = RECORDING_FIELDS["duration"].get_value(doc)

        if start_date and start_time:
            try:
                info.date = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass

        if duration_ms is not None:
            duration_s = int(duration_ms) // 1000
            info.duration = timedelta(seconds=duration_s)
            if info.date:
                info.end_date = info.date + info.duration

        return info

    def _read_leads(self, filepath: Path) -> list[Lead]:
        labels, waveform_data, sampling_freq, duration_ms = _read_sierra_leads(filepath)
        leads: list[Lead] = []
        for i, label in enumerate(labels):
            if i < len(waveform_data):
                samples = waveform_data[i].astype(np.float64)
                leads.append(Lead(
                    label=label,
                    samples=samples,
                    sample_rate=sampling_freq,
                ))
        return leads

    def _read_annotations(self, doc: dict) -> dict[str, str]:
        annotations: dict[str, str] = {}
        for field_name, xml_field in ANNOTATION_FIELDS.items():
            value = xml_field.get_value(doc)
            if value is not None:
                annotations[field_name] = str(value)

        statements = find_tag(doc, "statement")
        if isinstance(statements, list):
            for stmt in statements:
                if isinstance(stmt, dict):
                    code = stmt.get("statementcode", "")
                    annotations[f"statement_{code}_left"] = stmt.get("leftstatement", "")
                    annotations[f"statement_{code}_right"] = stmt.get("rightstatement", "")
        elif isinstance(statements, dict):
            code = statements.get("statementcode", "")
            annotations[f"statement_{code}_left"] = statements.get("leftstatement", "")
            annotations[f"statement_{code}_right"] = statements.get("rightstatement", "")

        return annotations

    def _read_device(self, doc: dict) -> DeviceInfo:
        info = DeviceInfo()

        machine_field = XMLField("dataacquisition", "machine/#text", "str")
        model = machine_field.get_value(doc)
        if model is not None:
            info.model = str(model)

        doc_ver_field = XMLField("documentinfo", "documentversion", "str")
        software_version = doc_ver_field.get_value(doc)
        if software_version is not None:
            info.software_version = str(software_version)

        machine_data = find_tag(doc, "machine")
        if isinstance(machine_data, dict):
            for key in ("manufacturer", "manufacturercode"):
                val = machine_data.get(key)
                if val and isinstance(val, str):
                    info.manufacturer = val
                    break

        acq_type_field = XMLField("signalcharacteristics", "acquisitiontype", "str")
        acq_type = acq_type_field.get_value(doc)
        if acq_type is not None:
            info.acquisition_type = str(acq_type)

        return info

    def _read_filters(self, doc: dict) -> FilterSettings:
        settings = FilterSettings()

        for fields_entry in SIGNAL_CHARACTERISTICS_FIELDS:
            for field_name, field_def in fields_entry.items():
                if not isinstance(field_def, XMLField):
                    continue
                value = field_def.get_value(doc)
                if value is None:
                    continue
                if field_name == "hipass" and settings.highpass is None:
                    settings.highpass = float(value)
                elif field_name == "lowpass" and settings.lowpass is None:
                    settings.lowpass = float(value)
                elif field_name == "notch_filter_freqs" and settings.notch is None:
                    settings.notch = float(int(value))
                elif field_name == "notch_filtered" and settings.notch_active is None:
                    settings.notch_active = bool(value)
                elif field_name == "artifact_filtered" and settings.artifact_filter is None:
                    settings.artifact_filter = bool(value)

        return settings

    def _read_measurements(self, doc: dict) -> GlobalMeasurements:
        measurements = GlobalMeasurements()

        def _safe_int(key: str) -> int | None:
            field = ANNOTATION_FIELDS.get(key)
            if field is None:
                return None
            val = field.get_value(doc)
            if val is None:
                return None
            try:
                return int(val)
            except (ValueError, TypeError):
                return None

        measurements.heart_rate = _safe_int("heartrate")
        measurements.rr_interval = _safe_int("rrint")
        measurements.pr_interval = _safe_int("print")
        measurements.qrs_duration = _safe_int("qrsdur")
        measurements.qt_interval = _safe_int("qtint")
        measurements.qtc_bazett = _safe_int("qtcb")
        measurements.qtc_fridericia = _safe_int("qtcf")
        measurements.p_axis = _safe_int("pfrontaxis")
        measurements.qrs_axis = _safe_int("qrsfrontaxis")
        measurements.t_axis = _safe_int("tfrontaxis")

        return measurements

    def _read_interpretation(self, doc: dict) -> Interpretation:
        interp = Interpretation()

        statements_data = find_tag(doc, "statement")
        stmt_texts: list[str] = []
        if isinstance(statements_data, list):
            for stmt in statements_data:
                if isinstance(stmt, dict):
                    left = stmt.get("leftstatement", "")
                    right = stmt.get("rightstatement", "")
                    text = f"{left} {right}".strip() if right else str(left).strip()
                    if text:
                        stmt_texts.append(text)
        elif isinstance(statements_data, dict):
            left = statements_data.get("leftstatement", "")
            right = statements_data.get("rightstatement", "")
            text = f"{left} {right}".strip() if right else str(left).strip()
            if text:
                stmt_texts.append(text)
        interp.statements = stmt_texts

        severity_code_field = ANNOTATION_FIELDS.get("severity_code")
        severity_field = ANNOTATION_FIELDS.get("severity")
        if severity_code_field is not None:
            sev_code = severity_code_field.get_value(doc)
            if sev_code is not None:
                interp.severity = str(sev_code)
        if not interp.severity and severity_field is not None:
            sev_text = severity_field.get_value(doc)
            if sev_text is not None:
                interp.severity = str(sev_text)

        interp_date_field = ANNOTATION_FIELDS.get("interpretation_date")
        interp_time_field = ANNOTATION_FIELDS.get("interpretation_time")
        if interp_date_field is not None and interp_time_field is not None:
            date_str = interp_date_field.get_value(doc)
            time_str = interp_time_field.get_value(doc)
            if date_str and time_str:
                try:
                    interp.interpretation_date = datetime.strptime(
                        f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S"
                    )
                except ValueError:
                    pass

        interp.source = "machine"

        md_sig_field = ANNOTATION_FIELDS.get("md_signature_line")
        if md_sig_field is not None:
            md_sig = md_sig_field.get_value(doc)
            if md_sig is not None:
                interp.interpreter = str(md_sig)

        return interp

    def _read_lead_measurements(self, doc: dict) -> dict | list | None:
        """Extract <leadmeasurements> section as raw data."""
        data = find_tag(doc, "leadmeasurements")
        return data

    def _read_order_info(self, doc: dict) -> dict | list | None:
        """Extract <orderinfo> section as raw data."""
        data = find_tag(doc, "orderinfo")
        return data

    def _read_signal(self, doc: dict) -> SignalCharacteristics:
        sig = SignalCharacteristics()
        for fields_entry in SIGNAL_CHARACTERISTICS_FIELDS:
            for field_name, field_def in fields_entry.items():
                if not isinstance(field_def, XMLField):
                    continue
                value = field_def.get_value(doc)
                if value is None:
                    continue
                if field_name == "bits_per_sample" and sig.bits_per_sample is None:
                    sig.bits_per_sample = int(value)
                elif field_name == "signal_offset" and sig.signal_offset is None:
                    sig.signal_offset = int(value)
                elif field_name == "signal_signed" and sig.signal_signed is None:
                    sig.signal_signed = bool(value)
                elif field_name == "number_channels_allocated" and sig.number_channels_allocated is None:
                    sig.number_channels_allocated = int(value)
                elif field_name == "number_channels_valid" and sig.number_channels_valid is None:
                    sig.number_channels_valid = int(value)
                elif field_name == "electrode_placement" and not sig.electrode_placement:
                    sig.electrode_placement = str(value)
                elif field_name == "acsetting" and sig.acsetting is None:
                    sig.acsetting = int(value)
                elif field_name == "compression" and not sig.compression:
                    sig.compression = str(value)
                elif field_name == "data_encoding" and not sig.data_encoding:
                    sig.data_encoding = str(value)
                elif field_name == "upsampled" and sig.upsampled is None:
                    sig.upsampled = bool(value)
                elif field_name == "upsampling_method" and not sig.upsampling_method:
                    sig.upsampling_method = str(value)
                elif field_name == "downsampled" and sig.downsampled is None:
                    sig.downsampled = bool(value)
                elif field_name == "waveform_modified" and sig.waveform_modified is None:
                    sig.waveform_modified = bool(value)
                elif field_name == "artifact_filtered" and sig.filtered is None:
                    sig.filtered = bool(value)
        return sig

    def _read_signal_characteristics(self, doc: dict) -> list[dict]:
        result: list[dict] = []
        for fields_entry in SIGNAL_CHARACTERISTICS_FIELDS:
            entry: dict = {}
            for field_name, field in fields_entry.items():
                if isinstance(field, XMLField):
                    entry[field_name] = field.get_value(doc)
                else:
                    entry[field_name] = field
            if entry.get("origin") is not None:
                result.append(entry)
        return result

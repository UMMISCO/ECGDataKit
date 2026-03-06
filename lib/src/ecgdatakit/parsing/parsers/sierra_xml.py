"""Philips Sierra ECG XML format parser.

Supports document types ``SierraECG`` and ``PhilipsECG`` versions
1.03, 1.04, 1.04.01 and 1.04.02.

The ``<signalcharacteristics><resolution>`` field stores **microvolts
per ADC count** (typically 5 µV).  Each :class:`~ecgdatakit.models.Lead`
is created with ``resolution=<uV/count>`` and ``units="uV"``.
``is_raw`` is auto-detected: ``True`` when resolution ≠ 1.0 (raw ADC
needing scaling), ``False`` when resolution = 1.0 (samples already in
physical units).
"""

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


# ── Severity code mapping ────────────────────────────────────────────
_SEVERITY_MAP: dict[str, str] = {
    "NM": "NORMAL",
    "ON": "OTHERWISE NORMAL",
    "BL": "BORDERLINE",
    "AB": "ABNORMAL",
}


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
    "age_default": XMLField("generalpatientdata", "age/@defaultage", "int"),
    "sex": XMLField("generalpatientdata", "sex", "str"),
    "weight": XMLField("generalpatientdata", "kg", "float"),
    "weight_lb": XMLField("generalpatientdata", "lb", "float"),
    "height": XMLField("generalpatientdata", "cm", "float"),
    "height_inch": XMLField("generalpatientdata", "inch", "float"),
    "race": XMLField("generalpatientdata", "race", "str"),
    "date_of_birth": XMLField("generalpatientdata", "dateofbirth", "str"),
    "pace_status": XMLField("generalpatientdata", "pacestatus", "str"),
    "unique_patient_id": XMLField("generalpatientdata", "uniquepatientid", "str"),
    "mrn": XMLField("generalpatientdata", "MRN", "str"),
}

RECORDING_FIELDS: dict[str, XMLField] = {
    "start_date": XMLField("dataacquisition", "@date", "str"),
    "start_time": XMLField("dataacquisition", "@time", "str"),
    "stat_flag": XMLField("dataacquisition", "@statflag", "bool"),
    "duration": XMLField("parsedwaveforms", "@durationperchannel", "int"),
}

ACQUIRER_FIELDS: dict[str, XMLField] = {
    "operator_id": XMLField("acquirer", "operator/@id", "str"),
    "department_id": XMLField("acquirer", "departmentid", "str"),
    "department_name": XMLField("acquirer", "departmentname", "str"),
    "institution_id": XMLField("acquirer", "institutionid", "str"),
    "institution_name": XMLField("acquirer", "institutionname", "str"),
    "facility_id": XMLField("acquirer", "facilityid", "str"),
    "facility_name": XMLField("acquirer", "facilityname", "str"),
    "room": XMLField("acquirer", "room", "str"),
    "bed": XMLField("acquirer", "bed", "str"),
    "encounter": XMLField("acquirer", "encounter", "str"),
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
    # globalmeasurements – values may carry an @editedflag attribute so
    # the text content lives under #text when parsed by xmltodict.
    "heartrate": XMLField("globalmeasurements", "heartrate/#text", "str"),
    "atrialrate": XMLField("globalmeasurements", "atrialrate/#text", "str"),
    "rrint": XMLField("globalmeasurements", "rrint/#text", "str"),
    "pdur": XMLField("globalmeasurements", "pdur/#text", "str"),
    "print": XMLField("globalmeasurements", "print/#text", "str"),
    "qonset": XMLField("globalmeasurements", "qonset/#text", "str"),
    "qrsdur": XMLField("globalmeasurements", "qrsdur/#text", "str"),
    "qtint": XMLField("globalmeasurements", "qtint/#text", "str"),
    "qtcb": XMLField("globalmeasurements", "qtcb/#text", "str"),
    "qtcf": XMLField("globalmeasurements", "qtcf/#text", "str"),
    # frontal-plane axes
    "pfrontaxis": XMLField("globalmeasurements", "pfrontaxis/#text", "str"),
    "i40frontaxis": XMLField("globalmeasurements", "i40frontaxis/#text", "str"),
    "t40frontaxis": XMLField("globalmeasurements", "t40frontaxis/#text", "str"),
    "qrsfrontaxis": XMLField("globalmeasurements", "qrsfrontaxis/#text", "str"),
    "stfrontaxis": XMLField("globalmeasurements", "stfrontaxis/#text", "str"),
    "tfrontaxis": XMLField("globalmeasurements", "tfrontaxis/#text", "str"),
    # horizontal-plane axes
    "phorizaxis": XMLField("globalmeasurements", "phorizaxis/#text", "str"),
    "i40horizaxis": XMLField("globalmeasurements", "i40horizaxis/#text", "str"),
    "t40horizaxis": XMLField("globalmeasurements", "t40horizaxis/#text", "str"),
    "qrshorizaxis": XMLField("globalmeasurements", "qrshorizaxis/#text", "str"),
    "sthorizaxis": XMLField("globalmeasurements", "sthorizaxis/#text", "str"),
    "thorizaxis": XMLField("globalmeasurements", "thorizaxis/#text", "str"),
    # interpretation metadata
    "md_signature_line": XMLField("interpretation", "mdsignatureline", "str"),
    "severity_code": XMLField("severity", "@code", "str"),
    "severity_id": XMLField("severity", "@id", "str"),
    "severity_text": XMLField("severity", "#text", "str"),
}


# ── DOM helpers (used by _read_sierra_leads for waveform decoding) ───

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


def _read_sierra_leads(filepath: Path) -> tuple[list[str], list[npt.NDArray[np.int16]], int, int, int]:
    """Read leads from a Sierra ECG file using DOM parsing.

    Returns
    -------
    labels : list[str]
        Lead names.
    sample_arrays : list[NDArray[int16]]
        Raw ADC arrays per lead.
    sampling_freq : int
        Sampling rate (Hz).
    duration_ms : int
        Duration per channel (ms).
    resolution_uv : int
        Resolution in microvolts per ADC count.
    """
    xdom = minidom.parse(str(filepath))
    root = _get_node(xdom, "restingecgdata")
    _assert_version(root)

    signal_details = _get_node(_get_node(root, "dataacquisition"), "signalcharacteristics")
    parsed_waveforms = _get_node(root, "parsedwaveforms")

    sampling_freq = int(_get_text(_get_node(signal_details, "samplingrate")))
    duration = int(_get_attr(parsed_waveforms, "durationperchannel"))

    # Resolution: uV per ADC count (typically 5)
    resolution_uv = int(_get_text(_get_node(signal_details, "resolution")))

    labels = _get_or_create_labels(signal_details, parsed_waveforms)
    waveform_data = _get_waveform_data(signal_details, parsed_waveforms, labels)

    # Einthoven / augmented limb lead corrections
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

    return labels, waveform_data, sampling_freq, duration, resolution_uv



class SierraXMLParser(Parser):
    """Parser for Philips Sierra ECG XML files.

    Supports both ``SierraECG`` and ``PhilipsECG`` document types
    (versions 1.03 – 1.04.02).  Waveform data may be uncompressed
    or XLI-compressed and is always Base64-encoded.

    Each lead carries ``resolution`` (µV per count) and ``units="uV"``.
    ``is_raw`` is auto-detected from the resolution metadata.
    """

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
        record.leads = self._read_leads(file_path, doc)
        record.median_beats = self._read_representative_beats(file_path)
        record.annotations = self._read_annotations(doc)
        record.recording.device = self._read_device(doc)
        record.recording.acquisition.filters = self._read_filters(doc)
        record.measurements = self._read_measurements(doc)
        record.interpretation = self._read_interpretation(doc)
        record.recording.acquisition.signal = self._read_signal(doc)

        # ── raw_metadata ────────────────────────────────────────
        record.raw_metadata["signal_characteristics"] = self._read_signal_characteristics(doc)
        record.raw_metadata["filepath"] = str(file_path)

        lead_measurements = self._read_lead_measurements(doc)
        if lead_measurements is not None:
            record.raw_metadata["lead_measurements"] = lead_measurements
            for lead in record.leads:
                if lead.label in lead_measurements:
                    lead.annotations = lead_measurements[lead.label]

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

        cross_lead = self._read_cross_lead_measurements(doc)
        if cross_lead is not None:
            record.raw_metadata["cross_lead_measurements"] = cross_lead

        group_meas = self._read_group_measurements(doc)
        if group_meas is not None:
            record.raw_metadata["group_measurements"] = group_meas

        config = self._read_config_settings(doc)
        if config:
            record.raw_metadata["config_settings"] = config

        user_defines = self._read_user_defines(doc)
        if user_defines:
            record.raw_metadata["user_defines"] = user_defines

        report_info = self._read_report_info(doc)
        if report_info:
            record.raw_metadata["report_info"] = report_info

        doc_info = self._read_document_info(doc)
        if doc_info:
            record.raw_metadata["document_info"] = doc_info

        # ── acquirer → recording fields ─────────────────────────
        for field_name, xml_field in ACQUIRER_FIELDS.items():
            value = xml_field.get_value(doc)
            if value is None:
                continue
            val_str = str(value).strip()
            if not val_str:
                continue
            if field_name == "operator_id" and not record.recording.technician:
                record.recording.technician = val_str
            elif field_name == "room" and not record.recording.room:
                record.recording.room = val_str
            elif field_name == "facility_name" and not record.recording.location:
                record.recording.location = val_str

        if record.leads:
            record.recording.acquisition.signal.sample_rate = record.leads[0].sample_rate

        return record

    # ── Patient ──────────────────────────────────────────────────

    def _read_patient(self, doc: dict) -> PatientInfo:
        info = PatientInfo()
        for field_name, xml_field in PATIENT_FIELDS.items():
            value = xml_field.get_value(doc)
            if value is None:
                continue
            if field_name == "patient_id":
                info.patient_id = str(value)
            elif field_name == "first_name":
                info.first_name = str(value)
            elif field_name == "last_name":
                info.last_name = str(value)
            elif field_name == "age":
                info.age = int(value)
            elif field_name == "age_default" and info.age is None:
                info.age = int(value)
            elif field_name == "sex":
                info.sex = str(value)
            elif field_name == "weight":
                info.weight = float(value)
            elif field_name == "weight_lb" and info.weight is None:
                # Convert lbs to kg
                info.weight = round(float(value) * 0.453592, 1)
            elif field_name == "height":
                info.height = float(value)
            elif field_name == "height_inch" and info.height is None:
                # Convert inches to cm
                info.height = round(float(value) * 2.54, 1)
            elif field_name == "race":
                info.race = str(value)
            elif field_name == "date_of_birth":
                try:
                    info.birth_date = datetime.strptime(str(value), "%Y-%m-%d")
                except ValueError:
                    pass
        return info

    # ── Recording ────────────────────────────────────────────────

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

    # ── Leads (with resolution & units for auto_scale) ───────────

    def _read_leads(self, filepath: Path, doc: dict) -> list[Lead]:
        labels, waveform_data, sampling_freq, duration_ms, resolution_uv = (
            _read_sierra_leads(filepath)
        )
        leads: list[Lead] = []
        for i, label in enumerate(labels):
            if i < len(waveform_data):
                samples = waveform_data[i].astype(np.float64)
                res = float(resolution_uv)
                raw = res != 1.0
                leads.append(Lead(
                    label=label,
                    samples=samples,
                    sample_rate=sampling_freq,
                    resolution=res,
                    resolution_unit="uV",
                    units="" if raw else "uV",
                    is_raw=raw,
                ))
        return leads

    # ── Representative / median beats ────────────────────────────

    def _read_representative_beats(self, filepath: Path) -> list[Lead]:
        """Parse representative beats from ``<repbeats>`` element."""
        try:
            xdom = minidom.parse(str(filepath))
            root = _get_node(xdom, "restingecgdata")
        except Exception:
            return []

        repbeats_node = _get_opt_node(root, "repbeats")
        if repbeats_node is None:
            return []

        try:
            signal_details = _get_node(
                _get_node(root, "dataacquisition"), "signalcharacteristics"
            )
            resolution_uv = int(_get_text(_get_node(signal_details, "resolution")))
        except Exception:
            return []

        # Sample rate: prefer repbeats attribute, fallback to rhythm
        rep_sr_str = _get_attr(repbeats_node, "samplespersecond", "0")
        rep_sr = int(rep_sr_str) if rep_sr_str != "0" else 0
        if rep_sr == 0:
            try:
                rep_sr = int(_get_text(_get_node(signal_details, "samplingrate")))
            except Exception:
                rep_sr = 500  # safe fallback

        # Labels
        try:
            labels = _get_or_create_labels(signal_details, repbeats_node)
        except Exception:
            return []

        # Decode waveform data
        try:
            waveform_data = _get_waveform_data(signal_details, repbeats_node, labels)
        except Exception:
            return []

        beats: list[Lead] = []
        for i, label in enumerate(labels):
            if i < len(waveform_data):
                samples = waveform_data[i].astype(np.float64)
                res = float(resolution_uv)
                raw = res != 1.0
                beats.append(Lead(
                    label=label,
                    samples=samples,
                    sample_rate=rep_sr,
                    resolution=res,
                    resolution_unit="uV",
                    units="" if raw else "uV",
                    is_raw=raw,
                ))
        return beats

    # ── Annotations ──────────────────────────────────────────────

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

    # ── Device ───────────────────────────────────────────────────

    def _read_device(self, doc: dict) -> DeviceInfo:
        info = DeviceInfo()

        # Model name (text content of <machine>)
        machine_field = XMLField("dataacquisition", "machine/#text", "str")
        model = machine_field.get_value(doc)
        if model is not None:
            info.model = str(model)

        # Device name / machine ID
        machine_id_field = XMLField("dataacquisition", "machine/@machineid", "str")
        machine_id = machine_id_field.get_value(doc)
        if machine_id is not None:
            info.name = str(machine_id)

        # Serial/detail description
        detail_field = XMLField("dataacquisition", "machine/@detaildescription", "str")
        detail = detail_field.get_value(doc)
        if detail is not None:
            info.serial_number = str(detail)
            # Extract manufacturer from detail (format: "Manufacturer:Serial:Version")
            parts = str(detail).split(":")
            if parts and not info.manufacturer:
                info.manufacturer = parts[0].strip()

        # Software version from document version
        doc_ver_field = XMLField("documentinfo", "documentversion", "str")
        software_version = doc_ver_field.get_value(doc)
        if software_version is not None:
            info.software_version = str(software_version)

        # Manufacturer fallback
        machine_data = find_tag(doc, "machine")
        if isinstance(machine_data, dict) and not info.manufacturer:
            for key in ("manufacturer", "manufacturercode"):
                val = machine_data.get(key)
                if val and isinstance(val, str):
                    info.manufacturer = val
                    break

        # Acquisition type
        acq_type_field = XMLField("signalcharacteristics", "acquisitiontype", "str")
        acq_type = acq_type_field.get_value(doc)
        if acq_type is not None:
            info.acquisition_type = str(acq_type)

        # Institution & department from acquirer
        for field_name, xml_field in ACQUIRER_FIELDS.items():
            value = xml_field.get_value(doc)
            if value is None:
                continue
            val_str = str(value).strip()
            if not val_str:
                continue
            if field_name == "institution_name" and not info.institution:
                info.institution = val_str
            elif field_name == "department_name" and not info.department:
                info.department = val_str

        return info

    # ── Filters ──────────────────────────────────────────────────

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

        # Also check reportbandwidth for filter settings
        rpt_hp = XMLField("reportbandwidth", "highpassfiltersetting", "float")
        rpt_lp = XMLField("reportbandwidth", "lowpassfiltersetting", "float")
        rpt_notch = XMLField("reportbandwidth", "notchfiltersetting", "float")
        rpt_art = XMLField("reportbandwidth", "artifactfilterflag", "bool")
        if settings.highpass is None:
            v = rpt_hp.get_value(doc)
            if v is not None:
                settings.highpass = float(v)
        if settings.lowpass is None:
            v = rpt_lp.get_value(doc)
            if v is not None:
                settings.lowpass = float(v)
        if settings.notch is None:
            v = rpt_notch.get_value(doc)
            if v is not None:
                settings.notch = float(v)
        if settings.artifact_filter is None:
            v = rpt_art.get_value(doc)
            if v is not None:
                settings.artifact_filter = bool(v)

        return settings

    # ── Global measurements ──────────────────────────────────────

    def _read_measurements(self, doc: dict) -> GlobalMeasurements:
        measurements = GlobalMeasurements()

        def _safe_int(key: str) -> int | None:
            f = ANNOTATION_FIELDS.get(key)
            if f is None:
                return None
            val = f.get_value(doc)
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

        # QRS count from crossleadmeasurements
        nqrs_field = XMLField("crossleadmeasurements", "numberofcomplexes", "int")
        nqrs = nqrs_field.get_value(doc)
        if nqrs is not None:
            measurements.qrs_count = int(nqrs)

        return measurements

    # ── Interpretation (statements as tuples, severity, clinician) ─

    def _read_interpretation(self, doc: dict) -> Interpretation:
        interp = Interpretation()

        # ── Statements as (left, right) tuples ───────────────────
        statements_data = find_tag(doc, "statement")
        stmt_tuples: list[tuple[str, str]] = []

        def _extract_stmt(stmt: dict) -> None:
            left = str(stmt.get("leftstatement", "")).strip()
            right = str(stmt.get("rightstatement", "")).strip()
            if left or right:
                stmt_tuples.append((left, right))

        if isinstance(statements_data, list):
            for stmt in statements_data:
                if isinstance(stmt, dict):
                    _extract_stmt(stmt)
        elif isinstance(statements_data, dict):
            _extract_stmt(statements_data)
        interp.statements = stmt_tuples

        # ── Severity ─────────────────────────────────────────────
        # <severity code="AB" id="4">- ABNORMAL ECG -</severity>
        sev_code = ANNOTATION_FIELDS["severity_code"].get_value(doc)
        if sev_code is not None:
            code = str(sev_code).strip().upper()
            interp.severity = _SEVERITY_MAP.get(code, code)
        if not interp.severity:
            sev_text = ANNOTATION_FIELDS["severity_text"].get_value(doc)
            if sev_text is not None:
                interp.severity = str(sev_text).strip().strip("-").strip()

        # ── Interpretation date ──────────────────────────────────
        date_str = ANNOTATION_FIELDS["interpretation_date"].get_value(doc)
        time_str = ANNOTATION_FIELDS["interpretation_time"].get_value(doc)
        if date_str and time_str:
            try:
                interp.interpretation_date = datetime.strptime(
                    f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                pass

        # ── Source ────────────────────────────────────────────────
        # Determine from document status and presence of clinician
        root_data = find_tag(doc, "restingecgdata")
        doc_status = ""
        if isinstance(root_data, dict):
            doc_status = str(root_data.get("@status", "")).strip().lower()

        confirming = find_tag(doc, "confirmingclinician")
        if confirming is not None and isinstance(confirming, dict):
            interp.source = "confirmed"
        elif doc_status == "confirmed":
            interp.source = "confirmed"
        else:
            interp.source = "machine"

        # ── Interpreter ──────────────────────────────────────────
        # MD signature line
        md_sig = ANNOTATION_FIELDS["md_signature_line"].get_value(doc)
        if md_sig is not None:
            interp.interpreter = str(md_sig).strip()

        # Fallback: confirming clinician name
        if not interp.interpreter and confirming is not None:
            if isinstance(confirming, dict):
                name = confirming.get("#text", "")
                if name:
                    interp.interpreter = str(name).strip()

        return interp

    # ── Lead measurements (per-lead morphology) ──────────────────

    def _read_lead_measurements(self, doc: dict) -> dict[str, dict[str, str]] | None:
        """Parse ``<leadmeasurements>`` into ``{lead_name: {field: value}}``."""
        raw = find_tag(doc, "leadmeasurement")
        if raw is None:
            return None
        items = raw if isinstance(raw, list) else [raw]
        result: dict[str, dict[str, str]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            lead_name = item.get("@leadname", "")
            if not lead_name:
                continue
            fields: dict[str, str] = {}
            # Collect attributes (flags like pmeasflag, qrsmeasflag, etc.)
            for k, v in item.items():
                if k.startswith("@") and k != "@leadname":
                    fields[k[1:]] = str(v)
            # Collect child elements (pamp, pdur, ramp, rdur, etc.)
            for k, v in item.items():
                if k.startswith("@") or k.startswith("#"):
                    continue
                if k in ("pacepulses", "leadqualitystates"):
                    continue  # complex sub-structures, skip
                if isinstance(v, dict):
                    text = v.get("#text", "")
                elif isinstance(v, str):
                    text = v
                else:
                    continue
                text = str(text).strip()
                if text:
                    fields[k] = text
            result[lead_name] = fields
        return result or None

    # ── Cross-lead measurements (VCG, axes, rhythm) ──────────────

    def _read_cross_lead_measurements(self, doc: dict) -> dict[str, str] | None:
        """Parse ``<crossleadmeasurements>`` into ``{field: value}``."""
        raw = find_tag(doc, "crossleadmeasurements")
        if not isinstance(raw, dict):
            return None
        result: dict[str, str] = {}
        # Collect attributes
        for k, v in raw.items():
            if k.startswith("@"):
                result[k[1:]] = str(v)
        # Collect child elements
        for k, v in raw.items():
            if k.startswith("@") or k.startswith("#"):
                continue
            if k in ("pacepulses", "qamessagecodes"):
                continue  # complex sub-structures
            if isinstance(v, dict):
                text = v.get("#text", "")
            elif isinstance(v, str):
                text = v
            else:
                continue
            text = str(text).strip()
            if text and text.lower() != "none":
                result[k] = text
        return result or None

    # ── Group measurements ───────────────────────────────────────

    def _read_group_measurements(self, doc: dict) -> list[dict[str, str]] | None:
        """Parse ``<groupmeasurements>`` into list of group dicts."""
        raw = find_tag(doc, "groupmeasurement")
        if raw is None:
            return None
        items = raw if isinstance(raw, list) else [raw]
        groups: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            fields: dict[str, str] = {}
            for k, v in item.items():
                if k.startswith("@"):
                    fields[k[1:]] = str(v)
                elif k.startswith("#"):
                    continue
                elif isinstance(v, dict):
                    text = v.get("#text", "")
                    if text and str(text).strip():
                        fields[k] = str(text).strip()
                elif isinstance(v, str) and v.strip():
                    fields[k] = v.strip()
            if fields:
                groups.append(fields)
        return groups or None

    # ── Order info ───────────────────────────────────────────────

    def _read_order_info(self, doc: dict) -> dict | list | None:
        """Extract ``<orderinfo>`` section as raw data."""
        return find_tag(doc, "orderinfo")

    # ── Config settings ──────────────────────────────────────────

    def _read_config_settings(self, doc: dict) -> dict | None:
        """Extract ``<configsettings>`` as a name→value dict."""
        data = find_tag(doc, "configsetting")
        if data is None:
            return None
        settings: dict[str, str] = {}
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                value = str(item.get("value", "")).strip()
                if name and name.lower() != "none":
                    settings[name] = value
        return settings or None

    # ── User defines ─────────────────────────────────────────────

    def _read_user_defines(self, doc: dict) -> dict | None:
        """Extract ``<userdefine>`` entries as a label→value dict."""
        data = find_tag(doc, "userdefine")
        if data is None:
            return None
        defines: dict[str, str] = {}
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict):
                label = str(item.get("label", "")).strip()
                value = str(item.get("value", "")).strip()
                if label and value and label.lower() != "none" and value.lower() != "none":
                    defines[label] = value
        return defines or None

    # ── Report info (gains, bandwidth) ───────────────────────────

    def _read_report_info(self, doc: dict) -> dict | None:
        """Extract ``<reportinfo>`` including gains and bandwidth."""
        result: dict[str, object] = {}

        # Report label & description
        rpt_label = XMLField("reportinfo", "reportlabel", "str")
        rpt_desc = XMLField("reportinfo", "reportdescription", "str")
        v = rpt_label.get_value(doc)
        if v is not None:
            result["report_label"] = str(v).strip()
        v = rpt_desc.get_value(doc)
        if v is not None:
            result["report_description"] = str(v).strip()

        # Amplitude gain
        gain_data = find_tag(doc, "amplitudegain")
        if isinstance(gain_data, dict):
            gain_info: dict[str, object] = {}
            unit = gain_data.get("@unit", "")
            if unit:
                gain_info["unit"] = str(unit)
            overall = gain_data.get("overallgain")
            if overall is not None:
                try:
                    gain_info["overall"] = float(str(overall).strip())
                except (ValueError, TypeError):
                    gain_info["overall"] = str(overall).strip()
            # Group gains (per-lead-group overrides)
            group = gain_data.get("groupgain")
            if group is not None:
                groups = group if isinstance(group, list) else [group]
                group_gains: list[dict[str, object]] = []
                for g in groups:
                    if isinstance(g, dict):
                        entry: dict[str, object] = {}
                        name = g.get("@leadgroupname", "")
                        if name:
                            entry["leads"] = str(name)
                        text = g.get("#text", "")
                        if text:
                            try:
                                entry["gain"] = float(str(text).strip())
                            except (ValueError, TypeError):
                                entry["gain"] = str(text).strip()
                        group_gains.append(entry)
                if group_gains:
                    gain_info["group_gains"] = group_gains
            if gain_info:
                result["amplitude_gain"] = gain_info

        # Time gain
        time_gain_data = find_tag(doc, "timegain")
        if time_gain_data is not None:
            tg_info: dict[str, object] = {}
            if isinstance(time_gain_data, dict):
                unit = time_gain_data.get("@unit", "")
                if unit:
                    tg_info["unit"] = str(unit)
                text = time_gain_data.get("#text", "")
                if text:
                    try:
                        tg_info["value"] = float(str(text).strip())
                    except (ValueError, TypeError):
                        tg_info["value"] = str(text).strip()
            else:
                try:
                    tg_info["value"] = float(str(time_gain_data).strip())
                except (ValueError, TypeError):
                    tg_info["value"] = str(time_gain_data).strip()
            if tg_info:
                result["time_gain"] = tg_info

        # Report bandwidth
        bw: dict[str, object] = {}
        for tag, key in [
            ("highpassfiltersetting", "highpass"),
            ("lowpassfiltersetting", "lowpass"),
            ("notchfiltersetting", "notch"),
            ("notchharmonicssetting", "notch_harmonics"),
            ("artifactfilterflag", "artifact_filter"),
            ("hysteresisfilterflag", "hysteresis_filter"),
        ]:
            f = XMLField("reportbandwidth", tag, "str")
            v = f.get_value(doc)
            if v is not None:
                bw[key] = str(v).strip()
        if bw:
            result["bandwidth"] = bw

        # Waveform format
        wf_fmt = find_tag(doc, "waveformformat")
        if isinstance(wf_fmt, dict):
            fmt: dict[str, object] = {}
            for attr in ("@leadsequence", "@timesequence"):
                val = wf_fmt.get(attr)
                if val:
                    fmt[attr.lstrip("@")] = str(val)
            main_wf = wf_fmt.get("mainwaveformformat")
            if isinstance(main_wf, dict):
                fmt["nrow"] = main_wf.get("@nrow", "")
                fmt["ncolumn"] = main_wf.get("@ncolumn", "")
                fmt["lead_layout"] = str(main_wf.get("#text", "")).strip()
            rhythm = wf_fmt.get("rhythmwaveformformat")
            if isinstance(rhythm, dict):
                fmt["rhythm_leads"] = str(rhythm.get("#text", "")).strip()
                fmt["nrhythm"] = rhythm.get("@nrhythm", "")
            if fmt:
                result["waveform_format"] = fmt

        return result or None

    # ── Document info ────────────────────────────────────────────

    def _read_document_info(self, doc: dict) -> dict | None:
        """Extract ``<documentinfo>`` metadata."""
        result: dict[str, str] = {}
        for tag in ("documentname", "filename", "documenttype",
                     "documentversion", "comments"):
            f = XMLField("documentinfo", tag, "str")
            v = f.get_value(doc)
            if v is not None:
                result[tag] = str(v).strip()

        editor = find_tag(doc, "editor")
        if isinstance(editor, dict):
            for attr in ("@date", "@time", "@id"):
                val = editor.get(attr)
                if val:
                    result[f"editor_{attr.lstrip('@')}"] = str(val)

        return result or None

    # ── Signal characteristics ───────────────────────────────────

    def _read_signal(self, doc: dict) -> SignalCharacteristics:
        sig = SignalCharacteristics()
        for fields_entry in SIGNAL_CHARACTERISTICS_FIELDS:
            for field_name, field_def in fields_entry.items():
                if not isinstance(field_def, XMLField):
                    continue
                value = field_def.get_value(doc)
                if value is None:
                    continue
                if field_name == "sampling_rate" and sig.sample_rate == 0:
                    sig.sample_rate = int(value)
                elif field_name == "resolution" and sig.resolution == 0.0:
                    sig.resolution = float(value)
                elif field_name == "bits_per_sample" and sig.bits_per_sample is None:
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

"""Philips Sierra ECG XML format parser.

Supports document types ``SierraECG`` and ``PhilipsECG`` versions
1.03, 1.04, 1.04.01 and 1.04.02 (PageWriter, TraceMasterVue and
IntelliSpace ECG exports, UTF-8 or UTF-16).

Rhythm waveforms (``<parsedwaveforms>``) are Base64 16-bit samples or
plain text, optionally XLI compressed.  In XLI data leads III, aVR, aVL
and aVF are stored as residuals and rebuilt from leads I and II.
Samples are kept as raw ADC counts with ``resolution`` in µV per count
(``<resolution>``, ``<signalresolution>`` in 1.03).  Representative
beats (``<repbeats>``) carry their own resolution and sampling rate.
"""

from __future__ import annotations

import binascii
import re
import warnings
from base64 import b64decode
from datetime import date, datetime, timedelta
from pathlib import Path
from xml.parsers.expat import ExpatError

import numpy as np
import numpy.typing as npt
import xmltodict

from ecgdatakit.exceptions import (
    CorruptedFileError,
    MissingElementError,
    UnsupportedFormatError,
)
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
from ecgdatakit.parsing.codecs.xli import xli_decode
from ecgdatakit.parsing.helpers import (
    STANDARD_LEADS,
    decode_text,
    fill_signal_summary,
    find_tag,
    normalize_lead_label,
    read_path,
    unique_labels,
)
from ecgdatakit.parsing.parser import Parser

_SUPPORTED_TYPES = {"SierraECG", "PhilipsECG"}
_SUPPORTED_VERSIONS = {"1.03", "1.04", "1.04.01", "1.04.02"}

_SEVERITY_MAP: dict[str, str] = {
    "NM": "NORMAL",
    "ON": "OTHERWISE NORMAL",
    "BL": "BORDERLINE",
    "BO": "BORDERLINE",
    "AB": "ABNORMAL",
}

_SEX_MAP = {"male": "M", "m": "M", "female": "F", "f": "F"}

# <pacestatus> values; "Unknown" and anything else leave has_pacemaker unset
_PACED = {"yes", "paced", "pacemaker", "true"}
_NOT_PACED = {"no", "notpaced", "not paced", "nopacemaker", "no pacemaker", "false"}

_LIMB_LEADS = list(STANDARD_LEADS[:6])

# Numeric placeholders in unparsed statements, e.g. "***" or "*.**"
_PLACEHOLDER = re.compile(r"\*+(?:\.\*+)?")
_SIGNATURE_DATE = re.compile(
    r"\s+(\d{1,2}-[A-Za-z]{3}-\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?"
    r"|\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?)\s*$"
)


# ── xmltodict access helpers ────────────────────────────────────────

def _first(node: object) -> object:
    if isinstance(node, list):
        return node[0] if node else None
    return node


def _as_list(node: object) -> list:
    if node is None:
        return []
    return node if isinstance(node, list) else [node]


def _get(node: object, *path: str) -> object:
    """Follow child element names, taking the first match at each level."""
    for key in path:
        node = _first(node)
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _text(node: object) -> str:
    """Stripped text content of an element ('' when absent)."""
    node = _first(node)
    if isinstance(node, dict):
        node = node.get("#text")
    if node is None:
        return ""
    return str(node).strip()


def _value(node: object) -> str:
    """Like :func:`_text` but treats the Philips placeholder ``---`` as empty."""
    text = _text(node)
    return "" if text.strip("- ") == "" else text


def _attr(node: object, name: str) -> str:
    node = _first(node)
    if isinstance(node, dict):
        value = node.get("@" + name)
        if value is not None:
            return str(value).strip()
    return ""


def _to_float(text: str) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _to_int(text: str) -> int | None:
    value = _to_float(text)
    return None if value is None else int(round(value))


def _to_bool(text: str) -> bool | None:
    key = text.strip().lower()
    if key in ("true", "yes", "t", "1"):
        return True
    if key in ("false", "no", "f", "0"):
        return False
    return None


def _parse_datetime(day: str, time: str) -> datetime | None:
    if not day or not time:
        return None
    try:
        return datetime.strptime(f"{day} {time}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _parse_date(text: str) -> datetime | None:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _age_at(birth: date, when: date) -> int:
    return when.year - birth.year - ((when.month, when.day) < (birth.month, birth.day))


def _flatten(node: object, skip: tuple[str, ...] = ()) -> dict[str, str]:
    """Attributes and simple child texts of an element as ``{name: text}``."""
    node = _first(node)
    result: dict[str, str] = {}
    if not isinstance(node, dict):
        return result
    for key, value in node.items():
        if key.startswith("#") or key in skip or key.lstrip("@") in skip:
            continue
        if key.startswith("@"):
            if key != "@xmlns" and str(value).strip():
                result[key[1:]] = str(value).strip()
            continue
        if isinstance(value, list) or (isinstance(value, dict) and any(
            not k.startswith(("@", "#")) for k in value
        )):
            continue
        text = _text(value)
        if text:
            result[key] = text
    return result


def _xmltodict(data: bytes | str) -> dict:
    try:
        return xmltodict.parse(data, disable_entities=True)
    except ValueError as e:  # raised by xmltodict for entity declarations
        raise CorruptedFileError(f"XML entity declarations are not allowed: {e}") from e


def _parse_xml(data: bytes) -> dict:
    """Parse the document, retrying with a lossless decode on encoding errors.

    Entity declarations are refused (xmltodict ``disable_entities``).
    """
    try:
        return _xmltodict(data)
    except ExpatError as first_error:
        # Undeclared non-UTF-8 bytes: decode without loss and drop the prolog
        try:
            text = decode_text(data)
            text = re.sub(r"^﻿?\s*<\?xml[^>]*\?>", "", text)
            return _xmltodict(text)
        except ExpatError:
            raise CorruptedFileError(f"Malformed XML: {first_error}") from first_error


# ── waveform helpers ────────────────────────────────────────────────

def _lead_name(lead_set: str, index: int) -> str:
    if lead_set in ("STD-12", "10-WIRE"):
        if 1 <= index <= 12:
            return STANDARD_LEADS[index - 1]
    return f"Channel {index}"


def _compression(waveform: object) -> str:
    """Compression method of a waveform element, '' when uncompressed."""
    flag = _attr(waveform, "compressflag")
    method = _attr(waveform, "compression") or _attr(waveform, "compressmethod")
    if flag and _to_bool(flag) is False:
        return ""
    if method.lower() in ("", "uncompressed", "none"):
        return ""
    return method


def _decode_base64(text: str, what: str) -> bytes:
    try:
        raw = b64decode("".join(text.split()), validate=True)
    except (binascii.Error, ValueError) as e:
        raise CorruptedFileError(f"{what}: invalid Base64 data ({e})") from e
    if not raw:
        raise CorruptedFileError(f"{what}: no waveform data")
    return raw


def _decode_plain(text: str, what: str) -> npt.NDArray[np.int32]:
    try:
        values = np.array([int(v) for v in text.split()], dtype=np.int32)
    except ValueError as e:
        raise CorruptedFileError(f"{what}: invalid plain sample value ({e})") from e
    if values.size == 0:
        raise CorruptedFileError(f"{what}: no waveform data")
    return values


def _rebuild_limb_leads(arrays: list[npt.NDArray[np.int32]]) -> None:
    """Rebuild III, aVR, aVL, aVF from the XLI residuals (in place)."""
    lead_i, lead_ii = arrays[0], arrays[1]
    lead_iii = lead_ii - lead_i - arrays[2]
    arrays[2] = lead_iii
    arrays[3] = -arrays[3] - (lead_i + lead_ii) // 2
    arrays[4] = (lead_i - lead_iii) // 2 - arrays[4]
    arrays[5] = (lead_ii + lead_iii) // 2 - arrays[5]


class SierraXMLParser(Parser):
    """Parser for Philips Sierra ECG XML files.

    Supports both ``SierraECG`` and ``PhilipsECG`` document types
    (versions 1.03 to 1.04.02).  Waveform data may be uncompressed or
    XLI-compressed, Base64 or plain text.

    Leads carry raw ADC counts with ``resolution`` in µV per count.
    """

    FORMAT_NAME = "Philips Sierra XML"
    FORMAT_DESCRIPTION = "Philips Sierra ECG XML format"
    FILE_EXTENSIONS = [".xml"]
    PRIORITY = 50

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        if b"<restingecgdata" in header:
            return True
        # UTF-16 exports (PageWriter TC, TraceMasterVue)
        tag = "<restingecgdata"
        return tag.encode("utf-16-le") in header or tag.encode("utf-16-be") in header

    def parse(self, file_path: Path) -> ECGRecord:
        path = Path(file_path)
        doc = _parse_xml(path.read_bytes())
        root = doc.get("restingecgdata") if isinstance(doc, dict) else None
        if not isinstance(root, dict):
            raise UnsupportedFormatError("Missing <restingecgdata> root element")
        doc_type, doc_ver = self._check_version(root)

        record = ECGRecord(source_format="sierra_xml")
        record.file_format = FileFormatInfo(version=doc_ver)
        record.raw_metadata["document_type"] = doc_type
        record.raw_metadata["filepath"] = str(path)

        acquisition = _get(root, "dataacquisition")
        signal_node = _get(acquisition, "signalcharacteristics")
        waveforms = _get(root, "waveforms")
        parsed = _get(waveforms, "parsedwaveforms")
        if parsed is None:
            parsed = find_tag(root, "parsedwaveforms")
        if parsed is None:
            raise MissingElementError("parsedwaveforms")
        interp_nodes = _as_list(_get(root, "interpretations", "interpretation"))
        if not interp_nodes:
            interp_nodes = _as_list(find_tag(root, "interpretation"))
        interp_node = interp_nodes[0] if interp_nodes else None
        meas_node = _get(root, "internalmeasurements") or _get(root, "measurements")

        record.recording = self._read_recording(acquisition, parsed, doc)
        record.patient = self._read_patient(root, record)
        record.leads = self._read_leads(parsed, signal_node)
        rhythm = record.leads[0] if record.leads else None
        record.median_beats = self._read_representative_beats(
            _get(waveforms, "repbeats"),
            rhythm.sampling_rate if rhythm else 0,
            rhythm.adc_resolution if rhythm else 0.0,
            record.raw_metadata,
        )
        record.recording.device = self._read_device(acquisition, signal_node)
        record.recording.acquisition.filters = self._read_filters(parsed, signal_node)
        record.recording.acquisition.signal = self._read_signal(parsed, signal_node, rhythm)
        record.measurements, qtc_extra = self._read_measurements(interp_node, meas_node)
        record.interpretation, machine_lines = self._read_interpretation(root, interp_node)
        record.annotations = self._read_annotations(interp_node)
        record.annotations.update(qtc_extra)
        if machine_lines and record.interpretation.source != "machine":
            record.annotations["machine_interpretation"] = "\n".join(machine_lines)

        self._read_raw_metadata(doc, root, record, interp_nodes, signal_node, parsed)
        fill_signal_summary(record)
        return record

    # ── Version ──────────────────────────────────────────────────

    @staticmethod
    def _check_version(root: dict) -> tuple[str, str]:
        doc_info = _get(root, "documentinfo")
        if doc_info is None:
            raise MissingElementError("documentinfo")
        doc_type = _text(_get(doc_info, "documenttype"))
        doc_ver = _text(_get(doc_info, "documentversion"))
        if not doc_type:
            raise MissingElementError("documenttype")
        if not doc_ver:
            raise MissingElementError("documentversion")
        if doc_type not in _SUPPORTED_TYPES or doc_ver not in _SUPPORTED_VERSIONS:
            raise UnsupportedFormatError(f"Files of type {doc_type} {doc_ver} are unsupported")
        return doc_type, doc_ver

    # ── Recording ────────────────────────────────────────────────

    def _read_recording(self, acquisition: object, parsed: object, doc: dict) -> RecordingInfo:
        info = RecordingInfo()
        info.date = _parse_datetime(_attr(acquisition, "date"), _attr(acquisition, "time"))

        duration_ms = _to_float(_attr(parsed, "durationperchannel"))
        if duration_ms is not None and duration_ms > 0:
            info.duration = timedelta(milliseconds=duration_ms)

        acquirer = _get(acquisition, "acquirer")
        operator = _get(acquirer, "operator")
        info.technician = (
            _value(_attr(operator, "id")) or _value(operator) or _value(_get(acquirer, "operatorid"))
        )
        for key in ("referringclinician", "orderingclinician"):
            name = _value(_get(acquirer, key))
            if name.strip(" ,.;"):
                info.referring_physician = name
                break
        if not info.referring_physician:
            order = _get(doc.get("restingecgdata"), "orderinfo")
            for key in ("referringphysician", "orderingphysician"):
                name = _value(_get(order, key))
                if name.strip(" ,.;"):
                    info.referring_physician = name
                    break
        info.room = _value(_get(acquirer, "room"))
        info.location = _value(_get(acquirer, "facilityname")) or _value(
            _get(acquirer, "institutionlocationid")
        )
        return info

    # ── Patient ──────────────────────────────────────────────────

    def _read_patient(self, root: dict, record: ECGRecord) -> PatientInfo:
        info = PatientInfo()
        data = _get(root, "patient", "generalpatientdata")
        if data is None:
            return info
        name = _get(data, "name")
        info.patient_id = _value(_get(data, "patientid")) or _value(_get(data, "MRN"))
        info.first_name = _value(_get(name, "firstname")) or _value(_get(data, "firstname"))
        info.last_name = _value(_get(name, "lastname")) or _value(_get(data, "lastname"))

        age = _get(data, "age")
        dob_text = _value(_get(age, "dateofbirth")) or _value(_get(data, "dateofbirth"))
        info.birth_date = _parse_date(dob_text) if dob_text else None
        years = _to_int(_value(_get(age, "years")))
        acquired = record.recording.date
        if years is not None:
            info.age = years
        elif info.birth_date is not None and acquired is not None:
            info.age = _age_at(info.birth_date.date(), acquired.date())
        # @defaultage is a device setting, never the patient's age
        default_age = _attr(age, "defaultage")
        if default_age:
            record.raw_metadata["default_age"] = default_age

        sex = _value(_get(data, "sex"))
        if sex:
            info.sex = _SEX_MAP.get(sex.lower(), "U")
        info.race = _value(_get(data, "race"))

        height = _get(data, "height")
        cm = _to_float(_value(_get(height, "cm")))
        inch = _to_float(_value(_get(height, "inch")))
        info.height = cm if cm is not None else (round(inch * 2.54, 1) if inch is not None else None)
        weight = _get(data, "weight")
        kg = _to_float(_value(_get(weight, "kg")))
        lb = _to_float(_value(_get(weight, "lb")))
        info.weight = kg if kg is not None else (round(lb * 0.45359237, 1) if lb is not None else None)

        pace = _value(_get(data, "pacestatus"))
        if pace:
            record.raw_metadata["pace_status"] = pace
            key = pace.lower()
            if key in _PACED:
                info.has_pacemaker = True
            elif key in _NOT_PACED:
                info.has_pacemaker = False

        for key in ("MRN", "uniquepatientid"):
            value = _value(_get(data, key))
            if value:
                record.raw_metadata[key.lower()] = value
        middle = _value(_get(name, "middlename"))
        if middle:
            record.raw_metadata["middle_name"] = middle
        medical = _get(root, "patient", "patientmedicaldata")
        if medical:
            record.raw_metadata["patient_medical_data"] = medical
        return info

    # ── Rhythm leads ─────────────────────────────────────────────

    @staticmethod
    def _resolution(parsed: object, signal_node: object) -> float | None:
        for text in (
            _attr(parsed, "resolution"),
            _text(_get(signal_node, "resolution")),
            _text(_get(signal_node, "signalresolution")),
        ):
            value = _to_float(text)
            if value is not None and value > 0:
                return value
        return None

    @staticmethod
    def _rhythm_labels(parsed: object, signal_node: object) -> list[str]:
        names = _attr(parsed, "leadlabels").split()
        if names:
            count = _to_int(_attr(parsed, "numberofleads"))
            if count is not None and 0 < count < len(names):
                names = names[:count]
        else:
            count = _to_int(_text(_get(signal_node, "numberchannelsallocated")))
            if count is None:
                count = _to_int(_text(_get(signal_node, "numberchannelsvalid")))
            if count is None or count <= 0:
                raise MissingElementError("numberchannelsallocated")
            lead_set = _text(_get(signal_node, "acquisitiontype")) or _text(
                _get(signal_node, "leadset")
            )
            names = [_lead_name(lead_set, k + 1) for k in range(count)]
        return unique_labels([normalize_lead_label(n) for n in names])

    def _read_leads(self, parsed: object, signal_node: object) -> list[Lead]:
        rate = _to_int(_attr(parsed, "samplespersecond")) or _to_int(
            _text(_get(signal_node, "samplingrate"))
        )
        if not rate or rate <= 0:
            raise CorruptedFileError("Missing or invalid sampling rate")
        duration = _to_float(_attr(parsed, "durationperchannel"))
        if duration is None or duration <= 0:
            raise CorruptedFileError("Missing or invalid parsedwaveforms@durationperchannel")
        n_samples = int(round(duration * rate / 1000))
        labels = self._rhythm_labels(parsed, signal_node)
        n_leads = len(labels)

        signed_text = _attr(parsed, "signalsigned") or _text(_get(signal_node, "signalsigned"))
        signed = _to_bool(signed_text) is not False
        encoding = _attr(parsed, "dataencoding")
        compression = _compression(parsed)
        text = _text(parsed)

        if compression:
            if compression.upper() != "XLI":
                raise UnsupportedFormatError(f"Compression unsupported: {compression}")
            if encoding != "Base64":
                raise UnsupportedFormatError(f"XLI data with encoding {encoding!r} is unsupported")
            chunks = xli_decode(_decode_base64(text, "parsedwaveforms"))
            if len(chunks) < n_leads:
                raise CorruptedFileError(
                    f"XLI data holds {len(chunks)} leads, {n_leads} expected"
                )
            arrays = []
            for label, chunk in zip(labels, chunks):
                if len(chunk) < n_samples:
                    raise CorruptedFileError(
                        f"Lead {label}: {len(chunk)} samples, {n_samples} expected"
                    )
                arrays.append(chunk[:n_samples].astype(np.int32))
            if labels[:6] == _LIMB_LEADS:
                _rebuild_limb_leads(arrays)
        else:
            if encoding == "Base64":
                raw = _decode_base64(text, "parsedwaveforms")
                if len(raw) < n_leads * n_samples * 2:
                    raise CorruptedFileError(
                        f"parsedwaveforms holds {len(raw)} bytes, "
                        f"{n_leads * n_samples * 2} expected"
                    )
                dtype = "<i2" if signed else "<u2"
                values = np.frombuffer(raw, dtype=dtype, count=n_leads * n_samples)
            elif encoding == "Plain":
                values = _decode_plain(text, "parsedwaveforms")
                if values.size < n_leads * n_samples:
                    raise CorruptedFileError(
                        f"parsedwaveforms holds {values.size} samples, "
                        f"{n_leads * n_samples} expected"
                    )
            else:
                raise UnsupportedFormatError(f"Waveform data encoding unsupported: {encoding}")
            values = values[: n_leads * n_samples].astype(np.int32)
            arrays = [values[k * n_samples:(k + 1) * n_samples] for k in range(n_leads)]

        resolution = self._resolution(parsed, signal_node)
        if resolution is None:
            warnings.warn(
                "Sierra XML: no signal resolution, leads are left as raw ADC counts",
                stacklevel=3,
            )
        # signaloffset is the ADC count that stands for 0 V
        zero = _to_int(_attr(parsed, "signaloffset") or _text(_get(signal_node, "signaloffset"))) or 0
        return [self._make_lead(label, arr, rate, resolution, zero) for label, arr in zip(labels, arrays)]

    @staticmethod
    def _make_lead(
        label: str,
        samples: npt.NDArray,
        rate: int,
        resolution: float | None,
        zero: int = 0,
    ) -> Lead:
        if resolution is None:
            return Lead(label=label, samples=samples.astype(np.float64), sampling_rate=rate)
        offset = -zero * resolution
        raw = derive_is_raw(resolution, offset, "uV")
        return Lead(
            label=label,
            samples=samples.astype(np.float64),
            sampling_rate=rate,
            resolution=resolution,
            resolution_unit="uV",
            offset=offset,
            units="" if raw else "uV",
            is_raw=raw,
            adc_resolution=resolution,
            adc_resolution_unit="uV",
        )

    # ── Representative beats ─────────────────────────────────────

    def _read_representative_beats(
        self, repbeats: object, rhythm_rate: int, rhythm_resolution: float, raw: dict,
    ) -> list[Lead]:
        """Decode ``<repbeats>``: one ``<repbeat leadname=...>`` per lead.

        1.04+ stores the samples in a ``<waveform duration=...>`` child,
        1.03 directly in ``<repbeat duration=...>``.
        """
        if repbeats is None:
            return []
        repbeats = _first(repbeats)
        compression = _attr(repbeats, "compression")
        if compression:
            warnings.warn(
                f"Sierra XML: representative beats compressed with {compression!r} "
                "are not supported and were skipped",
                stacklevel=3,
            )
            return []
        encoding = _attr(repbeats, "dataencoding")
        if encoding not in ("Base64", "Plain"):
            raise UnsupportedFormatError(
                f"Representative beat encoding unsupported: {encoding or 'missing'}"
            )
        rate = _to_int(_attr(repbeats, "samplespersec")) or _to_int(
            _attr(repbeats, "samplespersecond")
        )
        if not rate or rate <= 0:
            # Not stated for the beats (optional in 1.03): the rhythm rate is used and reported
            rate = rhythm_rate
            if not rate or rate <= 0:
                raise CorruptedFileError("Missing or invalid representative beat sampling rate")
            raw["median_sampling_rate_stated"] = False
            warnings.warn(
                f"Sierra XML: <repbeats> has no sampling rate, the rhythm rate ({rate} Hz) "
                "is used for the representative beats",
                stacklevel=3,
            )
        resolution = _to_float(_attr(repbeats, "resolution"))
        if resolution is None or resolution <= 0:
            resolution = rhythm_resolution or None
            raw["median_resolution_stated"] = False
            warnings.warn(
                "Sierra XML: <repbeats> has no resolution, "
                + (f"the rhythm resolution ({resolution} uV) is used for the representative beats"
                   if resolution else "the representative beats are left as raw ADC counts"),
                stacklevel=3,
            )

        items = _as_list(_get(repbeats, "repbeat"))
        names = unique_labels([normalize_lead_label(_attr(i, "leadname")) for i in items])
        beats: list[Lead] = []
        for label, item in zip(names, items):
            waveform = _get(item, "waveform")
            holder = waveform if waveform is not None else item
            what = f"repbeat {label}"
            if encoding == "Base64":
                raw = _decode_base64(_text(holder), what)
                if len(raw) % 2:
                    raise CorruptedFileError(f"{what}: odd byte count {len(raw)}")
                samples = np.frombuffer(raw, dtype="<i2").astype(np.int32)
            else:
                samples = _decode_plain(_text(holder), what)
            duration = _to_float(_attr(holder, "duration") or _attr(item, "duration"))
            if duration is not None and duration > 0:
                expected = int(round(duration * rate / 1000))
                if len(samples) < expected:
                    raise CorruptedFileError(
                        f"{what}: {len(samples)} samples, {expected} expected"
                    )
                samples = samples[:expected]
            beat = self._make_lead(label, samples, rate, resolution)
            beat.annotations = _flatten(item, skip=("leadname", "waveform"))
            beats.append(beat)

        method = _attr(repbeats, "repbeatmethod")
        if method:
            for beat in beats:
                beat.annotations.setdefault("repbeatmethod", method)
        return beats

    # ── Device ───────────────────────────────────────────────────

    def _read_device(self, acquisition: object, signal_node: object) -> DeviceInfo:
        info = DeviceInfo()
        machine = _get(acquisition, "machine")
        info.model = _value(machine)
        info.name = _value(_attr(machine, "machineid"))
        # detaildescription: "Manufacturer:product number:software version"
        parts = [p.strip() for p in _attr(machine, "detaildescription").split(":")]
        if parts and parts[0]:
            info.manufacturer = parts[0]
        if len(parts) >= 3 and parts[2]:
            info.software_version = ":".join(parts[2:])
        if not info.model and len(parts) >= 2:
            info.model = parts[1]

        acquirer = _get(acquisition, "acquirer")
        info.institution = _value(_get(acquirer, "institutionname"))
        info.department = _value(_get(acquirer, "departmentname"))
        info.acquisition_type = _text(_get(signal_node, "acquisitiontype"))
        return info

    # ── Filters ──────────────────────────────────────────────────

    def _read_filters(self, parsed: object, signal_node: object) -> FilterSettings:
        """Filters applied to the stored waveform.

        ``<parsedwaveforms>`` attributes describe the stored data (they differ
        from ``<signalcharacteristics>`` when the waveform was re-filtered).
        """
        settings = FilterSettings()

        def pick(name: str) -> str:
            return _attr(parsed, name) or _text(_get(signal_node, name))

        settings.highpass = _to_float(pick("hipass"))
        settings.lowpass = _to_float(pick("lowpass"))
        if settings.highpass is None and settings.lowpass is None:
            # 1.03: <signalbandwidth>0.05-150</signalbandwidth>
            band = _text(_get(signal_node, "signalbandwidth")).split("-")
            if len(band) == 2:
                settings.highpass = _to_float(band[0])
                settings.lowpass = _to_float(band[1])
        if settings.highpass == 0:
            settings.highpass = None
        if settings.lowpass == 0:
            settings.lowpass = None

        settings.notch_active = _to_bool(pick("notchfiltered"))
        notch = _to_float(pick("notchfilterfreqs"))
        settings.notch = notch if notch else None  # acsetting is the mains setting, not the notch
        settings.artifact_filter = _to_bool(_attr(parsed, "artfiltered"))
        return settings

    # ── Signal characteristics ───────────────────────────────────

    def _read_signal(
        self, parsed: object, signal_node: object, rhythm: Lead | None
    ) -> SignalCharacteristics:
        sig = SignalCharacteristics()
        if rhythm is not None:
            sig.sampling_rate = rhythm.sampling_rate
            if rhythm.resolution_unit:
                sig.resolution = rhythm.resolution

        def pick(name: str) -> str:
            return _attr(parsed, name) or _text(_get(signal_node, name))

        sig.bits_per_sample = _to_int(pick("bitspersample")) or _to_int(
            _attr(parsed, "nbitspersample")
        )
        sig.signal_offset = _to_int(pick("signaloffset"))
        sig.signal_signed = _to_bool(pick("signalsigned"))
        sig.number_channels_allocated = _to_int(_text(_get(signal_node, "numberchannelsallocated")))
        sig.number_channels_valid = _to_int(_text(_get(signal_node, "numberchannelsvalid")))
        sig.electrode_placement = _text(_get(signal_node, "electrodeplacement"))
        sig.acsetting = _to_int(_text(_get(signal_node, "acsetting")))
        sig.compression = _compression(parsed)
        sig.data_encoding = _attr(parsed, "dataencoding")
        sig.filtered = _to_bool(_attr(parsed, "filterflag"))
        sig.upsampled = _to_bool(_attr(parsed, "upsampled"))
        sig.upsampling_method = _attr(parsed, "upsamplemethod")
        sig.downsampled = _to_bool(_attr(parsed, "downsampled"))
        sig.downsampling_method = _attr(parsed, "downsamplemethod")
        sig.waveform_modified = _to_bool(_attr(parsed, "waveformmodified"))
        return sig

    # ── Global measurements ──────────────────────────────────────

    def _read_measurements(
        self, interp_node: object, meas_node: object,
    ) -> tuple[GlobalMeasurements, dict[str, str]]:
        """Global measurements in ms, bpm and degrees.

        1.04 keeps them in ``interpretation/globalmeasurements`` and
        ``internalmeasurements/crossleadmeasurements``; 1.03 in
        ``interpretation/interpretationmeasurements`` and
        ``measurements/globalmeasurements`` (``mean*`` names).
        """
        sources = [
            _get(interp_node, "globalmeasurements"),
            _get(interp_node, "interpretationmeasurements"),
            _get(meas_node, "crossleadmeasurements"),
            _get(meas_node, "globalmeasurements"),
        ]

        def pick(*keys: str) -> int | None:
            for node in sources:
                for key in keys:
                    value = _to_int(_text(_get(node, key)))
                    if value is not None:
                        return value
            return None

        m = GlobalMeasurements()
        m.heart_rate = pick("heartrate", "meanventrate")
        m.rr_interval = pick("rrint")
        m.pr_interval = pick("print", "meanprint")
        m.qrs_duration = pick("qrsdur", "meanqrsdur")
        m.qt_interval = pick("qtint", "meanqtint")
        m.qtc_bazett = pick("qtcb")
        m.qtc_fridericia = pick("qtcf")
        m.p_axis = pick("pfrontaxis")
        m.qrs_axis = pick("qrsfrontaxis")
        m.t_axis = pick("tfrontaxis")
        m.qrs_count = pick("numberofcomplexes")
        extra = {}
        mean_qtc = pick("meanqtc")
        if mean_qtc is not None:
            # 1.03 files do not state the QTc formula
            extra["qtc"] = f"{mean_qtc} ms (formula not stated)"
        return m, extra

    # ── Interpretation ───────────────────────────────────────────

    @staticmethod
    def _render_component(component: object) -> tuple[str, str]:
        """Render a coded statement template with its variables."""
        unparsed = _get(component, "unparsedstatement")
        if unparsed is None:
            return _text(component), ""
        variables = _get(component, "variables")
        numbers = [_text(v) for v in _as_list(_get(variables, "numericvalue"))]
        leads = " ".join(_text(v) for v in _as_list(_get(variables, "listofECGlead")))

        def fill(template: str) -> str:
            if leads:
                template = template.replace("*LEAD*", leads)
            values = iter(numbers)

            def sub(match: re.Match) -> str:
                value = next(values, None)
                return match.group(0) if value is None else value

            return " ".join(_PLACEHOLDER.sub(sub, template).split())

        modifiers = " ".join(
            _text(m) for m in _as_list(_get(component, "modifiers", "modifier")) if _text(m)
        )
        left = fill(_text(_get(unparsed, "lhsstatement")))
        right = fill(_text(_get(unparsed, "rhsstatement")))
        if modifiers:
            left = f"{modifiers} {left}".strip()
        return left, right

    def _read_interpretation(
        self, root: dict, interp_node: object
    ) -> tuple[Interpretation, list[str]]:
        """Final statements plus the device statements as text lines.

        ``status="Confirmed"`` or a ``<confirmingclinician>`` marks a
        confirmed ECG; edited but unconfirmed statements are an overread.
        """
        interp = Interpretation()
        if interp_node is None:
            return interp, []

        final: list[tuple[str, str]] = []
        edited = False
        for stmt in _as_list(_get(interp_node, "statement")):
            left = _text(_get(stmt, "leftstatement"))
            right = _text(_get(stmt, "rightstatement"))
            if _to_bool(_attr(stmt, "editedflag")):
                edited = True
            if left or right:
                final.append((left, right))

        machine: list[tuple[str, str]] = []
        components = _first(_get(interp_node, "interpretationdatastructure", "statementcomponents"))
        if isinstance(components, dict):
            for key, value in components.items():
                if key.startswith(("@", "#")):
                    continue
                for comp in _as_list(value):
                    source = _attr(comp, "source").lower()
                    if source == "editor" or _to_bool(_attr(comp, "deleted")):
                        edited = True
                    if source != "editor":
                        left, right = self._render_component(comp)
                        if left or right:
                            machine.append((left, right))
        if not machine and not edited:
            machine = list(final)

        severity = _get(interp_node, "severity")
        code = _attr(severity, "code").upper()
        interp.severity = _SEVERITY_MAP.get(code, "") or _text(severity).strip("- ").strip()

        status = _attr(root, "status").lower()
        confirmer = _get(interp_node, "confirmingclinician")
        signature = _text(_get(interp_node, "mdsignatureline"))
        if status == "confirmed" or _value(confirmer):
            interp.source = "confirmed"
            interp.interpreter = _value(confirmer)
            interp.interpretation_date = _parse_datetime(
                _attr(confirmer, "date"), _attr(confirmer, "time")
            )
            if signature and (not interp.interpreter or interp.interpretation_date is None):
                name, signed_at = self._parse_signature(signature)
                interp.interpreter = interp.interpreter or name
                interp.interpretation_date = interp.interpretation_date or signed_at
        elif edited:
            interp.source = "overread"
            editor = _get(root, "documentinfo", "editor")
            interp.interpreter = _value(editor)
            interp.interpretation_date = _parse_datetime(_attr(editor, "date"), _attr(editor, "time"))
        else:
            interp.source = "machine"
            interp.interpretation_date = _parse_datetime(
                _attr(interp_node, "date"), _attr(interp_node, "time")
            )

        interp.statements = final
        lines = [f"{left} | {right}" if right else left for left, right in machine]
        return interp, lines

    @staticmethod
    def _parse_signature(signature: str) -> tuple[str, datetime | None]:
        """Split ``"Confirmed by: NAME 24-Dec-2008 07:15:13"`` into name and date."""
        text = signature
        prefix, sep, rest = signature.partition(":")
        # Drop a leading "Confirmed by:" style prefix (no digits in it)
        if sep and not any(c.isdigit() for c in prefix):
            text = rest.strip()
        signed_at = None
        match = _SIGNATURE_DATE.search(text)
        if match:
            stamp = " ".join(match.group(1).split())
            for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"):
                try:
                    signed_at = datetime.strptime(stamp, fmt)
                    break
                except ValueError:
                    continue
            text = text[: match.start()].strip()
        return text, signed_at

    # ── Annotations ──────────────────────────────────────────────

    def _read_annotations(self, interp_node: object) -> dict[str, str]:
        annotations: dict[str, str] = {}
        if interp_node is None:
            return annotations
        for key in ("criteriaversion", "criteriaversiondate", "customcriteriaversion"):
            value = _attr(interp_node, key)
            if value:
                annotations[key] = value
        for node in (
            _get(interp_node, "globalmeasurements"),
            _get(interp_node, "interpretationmeasurements"),
        ):
            for key, value in _flatten(node).items():
                if key not in ("editedflag", "arrhyflag"):
                    annotations.setdefault(key, value)
        code = _attr(_get(interp_node, "severity"), "code")
        if code:
            annotations["severity_code"] = code
        return annotations

    # ── raw_metadata ─────────────────────────────────────────────

    def _read_raw_metadata(
        self,
        doc: dict,
        root: dict,
        record: ECGRecord,
        interp_nodes: list,
        signal_node: object,
        parsed: object,
    ) -> None:
        meta = record.raw_metadata
        acquisition = _get(root, "dataacquisition")
        if _attr(acquisition, "date") and record.recording.date is None:
            meta["acquisition_date"] = _attr(acquisition, "date")
        stat = _to_bool(_attr(acquisition, "statflag"))
        if stat is not None:
            meta["stat_flag"] = stat
        meta["signal_characteristics"] = _flatten(signal_node)
        meta["parsed_waveforms"] = _flatten(parsed)
        machine = _get(acquisition, "machine")
        parts = _attr(machine, "detaildescription").split(":")
        if len(parts) >= 2 and parts[1].strip():
            meta["device_product_number"] = parts[1].strip()
        acquirer = _get(acquisition, "acquirer")
        if acquirer:
            meta["acquirer"] = acquirer
        if root.get("@status"):
            meta["status"] = str(root["@status"])

        statements = []
        for interp_node in interp_nodes:
            for stmt in _as_list(_get(interp_node, "statement")):
                statements.append({
                    "code": _text(_get(stmt, "statementcode")),
                    "left": _text(_get(stmt, "leftstatement")),
                    "right": _text(_get(stmt, "rightstatement")),
                    "edited": _to_bool(_attr(stmt, "editedflag")),
                })
        if statements:
            meta["statements"] = statements
        signature = _text(_get(interp_nodes[0], "mdsignatureline")) if interp_nodes else ""
        if signature:
            meta["md_signature_line"] = signature
        if len(interp_nodes) > 1:
            meta["other_interpretations"] = interp_nodes[1:]

        lead_measurements = self._read_lead_measurements(doc)
        if lead_measurements:
            meta["lead_measurements"] = lead_measurements
            for lead in record.leads:
                if lead.label in lead_measurements:
                    lead.annotations = lead_measurements[lead.label]
        for key, value in (
            ("order_info", find_tag(doc, "orderinfo")),
            ("cross_lead_measurements", self._read_cross_lead_measurements(doc)),
            ("group_measurements", self._read_group_measurements(doc)),
            ("config_settings", self._read_config_settings(doc)),
            ("user_defines", self._read_user_defines(doc)),
            ("report_info", self._read_report_info(doc)),
            ("document_info", self._read_document_info(doc)),
        ):
            if value:
                meta[key] = value

    def _read_lead_measurements(self, doc: dict) -> dict[str, dict[str, str]] | None:
        """Parse ``<leadmeasurements>`` into ``{lead_name: {field: value}}``."""
        result: dict[str, dict[str, str]] = {}
        for item in _as_list(find_tag(doc, "leadmeasurement")):
            name = _attr(item, "leadname")
            if name:
                label = normalize_lead_label(name)
                result[label] = _flatten(item, skip=("leadname", "pacepulses", "leadqualitystates"))
        return result or None

    def _read_cross_lead_measurements(self, doc: dict) -> dict[str, str] | None:
        """Parse ``<crossleadmeasurements>`` into ``{field: value}``."""
        result = _flatten(find_tag(doc, "crossleadmeasurements"), skip=("pacepulses", "qamessagecodes"))
        return {k: v for k, v in result.items() if v.lower() != "none"} or None

    def _read_group_measurements(self, doc: dict) -> list[dict[str, str]] | None:
        """Parse ``<groupmeasurements>`` into a list of group dicts."""
        groups = [_flatten(item) for item in _as_list(find_tag(doc, "groupmeasurement"))]
        return [g for g in groups if g] or None

    def _read_config_settings(self, doc: dict) -> dict[str, str] | None:
        """Extract ``<configsettings>`` as a name to value dict."""
        settings: dict[str, str] = {}
        for item in _as_list(find_tag(doc, "configsetting")):
            name = _text(_get(item, "name"))
            if name and name.lower() != "none":
                settings[name] = _text(_get(item, "value"))
        return settings or None

    def _read_user_defines(self, doc: dict) -> dict[str, str] | None:
        """Extract ``<userdefine>`` entries as a label to value dict."""
        defines: dict[str, str] = {}
        for item in _as_list(find_tag(doc, "userdefine")):
            label = _text(_get(item, "label"))
            value = _text(_get(item, "value"))
            if label and value and label.lower() != "none" and value.lower() != "none":
                defines[label] = value
        return defines or None

    def _read_report_info(self, doc: dict) -> dict | None:
        """Extract ``<reportinfo>`` including gains and display bandwidth."""
        info = find_tag(doc, "reportinfo")
        result: dict[str, object] = {}
        for key, name in (("reportlabel", "report_label"), ("reportdescription", "report_description"),
                          ("reporttype", "report_type")):
            value = _text(_get(info, key))
            if value:
                result[name] = value

        gain = find_tag(doc, "amplitudegain")
        if isinstance(gain, dict):
            gain_info: dict[str, object] = {}
            if _attr(gain, "unit"):
                gain_info["unit"] = _attr(gain, "unit")
            overall = _text(gain.get("overallgain"))
            if overall:
                gain_info["overall"] = _to_float(overall) if _to_float(overall) is not None else overall
            group_gains = []
            for g in _as_list(gain.get("groupgain")):
                entry: dict[str, object] = {}
                if _attr(g, "leadgroupname"):
                    entry["leads"] = _attr(g, "leadgroupname")
                text = _text(g)
                if text:
                    entry["gain"] = _to_float(text) if _to_float(text) is not None else text
                group_gains.append(entry)
            if group_gains:
                gain_info["group_gains"] = group_gains
            if gain_info:
                result["amplitude_gain"] = gain_info

        time_gain = find_tag(doc, "timegain")
        if time_gain is not None and not isinstance(time_gain, list):
            tg: dict[str, object] = {}
            if _attr(time_gain, "unit"):
                tg["unit"] = _attr(time_gain, "unit")
            text = _text(time_gain)
            if text:
                tg["value"] = _to_float(text) if _to_float(text) is not None else text
            if tg:
                result["time_gain"] = tg

        bandwidth: dict[str, object] = {}
        bw_node = find_tag(doc, "reportbandwidth")
        for tag, key in [
            ("highpassfiltersetting", "highpass"),
            ("lowpassfiltersetting", "lowpass"),
            ("notchfiltersetting", "notch"),
            ("notchharmonicssetting", "notch_harmonics"),
            ("artifactfilterflag", "artifact_filter"),
            ("hysteresisfilterflag", "hysteresis_filter"),
        ]:
            value = _text(_get(bw_node, tag))
            if value:
                bandwidth[key] = value
        if bandwidth:
            result["bandwidth"] = bandwidth

        wf_fmt = find_tag(doc, "waveformformat")
        if isinstance(wf_fmt, dict):
            fmt: dict[str, object] = {}
            for name in ("leadsequence", "timesequence"):
                if _attr(wf_fmt, name):
                    fmt[name] = _attr(wf_fmt, name)
            main_wf = wf_fmt.get("mainwaveformformat")
            if main_wf is not None:
                fmt["nrow"] = _attr(main_wf, "nrow")
                fmt["ncolumn"] = _attr(main_wf, "ncolumn")
                fmt["lead_layout"] = _text(main_wf)
            rhythm = wf_fmt.get("rhythmwaveformformat")
            if rhythm is not None:
                fmt["rhythm_leads"] = _text(rhythm)
                fmt["nrhythm"] = _attr(rhythm, "nrhythm")
            if fmt:
                result["waveform_format"] = fmt
        return result or None

    def _read_document_info(self, doc: dict) -> dict[str, str] | None:
        """Extract ``<documentinfo>`` metadata."""
        info = find_tag(doc, "documentinfo")
        result: dict[str, str] = {}
        for tag in ("documentname", "filename", "documenttype", "documentversion", "comments"):
            value = _text(read_path(info, tag) if isinstance(info, dict) else None)
            if value:
                result[tag] = value
        editor = _get(info, "editor")
        for name in ("date", "time", "id"):
            if _attr(editor, name):
                result[f"editor_{name}"] = _attr(editor, name)
        if _value(editor):
            result["editor"] = _value(editor)
        return result or None

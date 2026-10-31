"""HL7 Annotated ECG (aECG) exporter.

Writes an :class:`~ecgdatakit.models.ECGRecord` as an HL7 aECG XML document
(ANSI/HL7 V3 ECG R1-2004, message type ``PORT_MT020001``), the format of the
FDA ECG warehouse.

The HL7 schema is not shipped with ECGDataKit. :func:`fetch_hl7_aecg_schema`
downloads it (release 2003-12, as distributed by the FDA) into a cache
folder, checking every file against a known SHA-256 hash, and
:func:`validate_hl7_aecg` checks a document against it. The exporter checks
every document it writes when created with ``validate=True``.

What is written:

* the rhythm leads, as a ``RHYTHM`` series: one sequence set per sampling
  rate and length, each with its time sequence (``TIME_ABSOLUTE`` from the
  recording date, ``TIME_RELATIVE`` without one) and one ``SLIST_PQ``
  sequence per lead, coded with the MDC lead codes of ISO/IEEE 11073;
* the representative beats, as a ``REPRESENTATIVE_BEAT`` series derived
  from the rhythm series, with the wave boundaries found in
  ``record.annotations`` (``representative_beat_p_onset_ms`` and so on);
* the patient (ID, name, sex, birth date, age, clinical history), the
  device (manufacturer, model, software, serial number), the technician,
  the filters, the global measurements, the interpretation statements and
  the rhythm statement.

Samples are written as integers (``digits``) with an origin and a scale in
µV. A lead whose values are a whole number of ADC counts, or of a fraction
of a count up to 1/12 (Wilson correction, computed leads), is written
exactly. Other values, such as averaged beats, are rounded to 0.01 µV, or
to 1/100 of the ADC resolution when that is finer, and a warning gives the
largest rounding error.

The schema requires a clinical trial context: a subject ID and a trial ID.
The subject ID is the patient ID, the trial ID is *trial_id*. Both are
scoped by *id_root*, an OID or UUID naming the organisation that assigned
them. Without a value, an ID is written with ``nullFlavor="NI"``.
"""

from __future__ import annotations

import functools
import hashlib
import os
import re
import urllib.request
import uuid
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ecgdatakit.exceptions import ExportError
from ecgdatakit.exporter.base import Exporter
from ecgdatakit.models import _TO_UV, ECGRecord, Lead, _normalize_unit

_NS = "urn:hl7-org:v3"
_XSI = "http://www.w3.org/2001/XMLSchema-instance"
# Elements are written in the HL7 namespace without a prefix
ET.register_namespace("", _NS)
ET.register_namespace("xsi", _XSI)

_ACT_CODE = "2.16.840.1.113883.5.4"
_MDC = "2.16.840.1.113883.6.24"
_CPT4 = "2.16.840.1.113883.6.12"
_LOINC = "2.16.840.1.113883.6.1"
_GENDER = "2.16.840.1.113883.5.1"

DEFAULT_ID_ROOT = str(uuid.uuid5(uuid.NAMESPACE_URL, "https://ecgdatakit.ummisco.fr/hl7-aecg/ids"))
"""UUID scoping the patient, trial and device IDs when no *id_root* is given."""

# HL7 aECG schema, release 2003-12, as distributed by the FDA. The files are
# downloaded from a fixed commit and must match these hashes.
_SCHEMA_URL = ("https://raw.githubusercontent.com/FDA/aecg-python/"
               "b0437290fbf6ded52011fa77db4b258a9bcbf637/src/aecg/data/hl7/2003-12%20Schema/")
_SCHEMA_FILES = {
    "schema/PORT_MT020001.xsd": "c4cea774858f94094f33fd51452c724385eac1717fc073cb2ff268e29b14c858",
    "dt/datatypes.xsd": "9c346352834c61a3b325e0413919be45b986d0fbd093725a60325ef33118abae",
    "voc/voc.xsd": "237fad20e4d7d500b47ea64621375ab297d31967d4feb49cd06a4939abf639d9",
    "meta/voc.xsd": "b26bc09c24930e0f49d9778e740486e74d1071b9ac7c7afb3418dba33908d438",
}
_SCHEMA_MAIN = "schema/PORT_MT020001.xsd"

# MDC lead codes (ISO/IEEE 11073-10101), by the suffix after MDC_ECG_LEAD_
_MDC_LEADS = (
    "I", "II", "III", "AVR", "AVL", "AVF", "AVRneg",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V2R", "V3R", "V4R", "V5R", "V6R", "V7R", "V8R", "V9R",
    "X", "Y", "Z", "CC5", "CM5", "LA", "RA", "LL",
    "fI", "fE", "fC", "fA", "fM", "fF", "fH", "D", "A", "J",
)
_LEAD_ALIASES = {"aVR": "AVR", "aVL": "AVL", "aVF": "AVF", "-aVR": "AVRneg"}
_MDC_LEADS_UPPER = {code.upper(): code for code in _MDC_LEADS}

# GlobalMeasurements field: (MDC code, unit)
_MEASUREMENTS = (
    ("heart_rate", "MDC_ECG_HEART_RATE", "bpm"),
    ("rr_interval", "MDC_ECG_TIME_PD_RR_GL", "ms"),
    ("pr_interval", "MDC_ECG_TIME_PD_PQ_GL", "ms"),
    ("qrs_duration", "MDC_ECG_TIME_PD_QRS_GL", "ms"),
    ("qt_interval", "MDC_ECG_TIME_PD_QT_GL", "ms"),
    ("qtc_bazett", "MDC_ECG_TIME_PD_QTC_BAZETT", "ms"),
    ("qtc_fridericia", "MDC_ECG_TIME_PD_QTC_FREDERICA", "ms"),
    ("p_axis", "MDC_ECG_ANGLE_P_FRONT", "deg"),
    ("qrs_axis", "MDC_ECG_ANGLE_QRS_FRONT", "deg"),
    ("t_axis", "MDC_ECG_ANGLE_T_FRONT", "deg"),
)

# Representative beat waves: (annotation key prefix, MDC wave code)
_WAVES = (
    ("representative_beat_p", "MDC_ECG_WAVC_PWAVE"),
    ("representative_beat_qrs", "MDC_ECG_WAVC_QRSWAVE"),
    ("representative_beat_t", "MDC_ECG_WAVC_TWAVE"),
)

# Fractions of an ADC count tried before rounding (Wilson correction gives
# thirds, Goldberger leads halves)
_DIVISORS = (1, 2, 3, 4, 6, 12)
_INTEGER_TOLERANCE = 1e-6
# Largest rounding step (µV) of values that are not on the ADC lattice
_MAX_STEP_UV = 0.01


# Characters XML 1.0 does not allow (control characters, surrogates, U+FFFE/F)
_NOT_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff￾￿]")


def _tag(name: str) -> str:
    return f"{{{_NS}}}{name}"


def _xml_text(value: str) -> str:
    """*value* without the characters XML cannot hold (binary formats can
    store control characters in their text fields)."""
    return _NOT_XML.sub("", value)


def _el(parent: ET.Element, name: str, text: str | None = None, **attrs: str) -> ET.Element:
    """Child *name* of *parent*, attributes given as keyword arguments
    (``xsi_type`` for ``xsi:type``)."""
    el = ET.SubElement(parent, _tag(name))
    for key, value in attrs.items():
        if key == "xsi_type":
            el.set(f"{{{_XSI}}}type", value)
        else:
            el.set(key, _xml_text(value))
    if text is not None:
        el.text = _xml_text(text)
    return el


def _number(value: float) -> str:
    """Shortest decimal text that gives *value* back."""
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def _ts(value: datetime) -> str:
    """HL7 TS value of *value* (``YYYYMMDDHHMMSS[.fff][+ZZzz]``)."""
    text = value.strftime("%Y%m%d%H%M%S")
    if value.microsecond:
        text += f".{value.microsecond:06d}".rstrip("0")
    if value.utcoffset() is not None:
        text += value.strftime("%z")
    return text


def _lead_code(label: str) -> str | None:
    """MDC code of the lead *label*, or ``None`` when MDC has none."""
    suffix = _LEAD_ALIASES.get(label) or _MDC_LEADS_UPPER.get(label.upper())
    return f"MDC_ECG_LEAD_{suffix}" if suffix else None


def _physical_uv(lead: Lead) -> np.ndarray:
    """Samples of *lead* in µV."""
    if lead.is_raw:
        unit = _normalize_unit(lead.resolution_unit) if lead.resolution_unit else None
        if unit is None or lead.resolution == 0.0:
            raise ExportError(
                f"Lead '{lead.label}' has no voltage scale (uncalibrated counts): "
                "HL7 aECG stores voltages"
            )
        values = (np.asarray(lead.samples, dtype=np.float64) * lead.resolution
                  + lead.offset) * _TO_UV[unit]
    else:
        unit = _normalize_unit(lead.units) if lead.units else None
        if unit is None:
            raise ExportError(
                f"Lead '{lead.label}' is in {lead.units or 'no unit'!r}, not a voltage: "
                "HL7 aECG stores voltages"
            )
        values = np.asarray(lead.samples, dtype=np.float64) * _TO_UV[unit]
    if not np.all(np.isfinite(values)):
        raise ExportError(f"Lead '{lead.label}' has missing (NaN) or infinite samples")
    return values


def _lattice(lead: Lead) -> tuple[float, float] | None:
    """ADC resolution and offset of *lead* in µV, when the file gave them."""
    unit = _normalize_unit(lead.resolution_unit) if lead.resolution_unit else None
    if unit is None or not lead.resolution:
        return None
    factor = _TO_UV[unit]
    return abs(lead.resolution) * factor, lead.offset * factor


def _digits(lead: Lead, step_uv: float | None) -> tuple[np.ndarray, float, float, float]:
    """``(digits, scale, origin, largest error)`` in µV for *lead*.

    The values are written exactly when they are a whole number of ADC
    counts or of a fraction of a count, otherwise rounded to *step_uv*.
    """
    values = _physical_uv(lead)
    lattice = _lattice(lead)
    origin = 0.0
    if lattice is not None and step_uv is None:
        base, origin = lattice
        for divisor in _DIVISORS:
            scale = base / divisor
            counts = (values - origin) / scale
            digits = np.rint(counts)
            if not len(counts) or np.max(np.abs(counts - digits)) <= _INTEGER_TOLERANCE:
                return digits.astype(np.int64), scale, origin, 0.0
    if step_uv is None:
        step_uv = min(lattice[0] / 100, _MAX_STEP_UV) if lattice is not None else _MAX_STEP_UV
    digits = np.rint((values - origin) / step_uv)
    error = float(np.max(np.abs(digits * step_uv + origin - values))) if len(values) else 0.0
    return digits.astype(np.int64), step_uv, origin, error


def default_schema_dir() -> Path:
    """Folder where :func:`fetch_hl7_aecg_schema` keeps the HL7 schema.

    ``$ECGDATAKIT_CACHE/hl7_aecg_schema`` when ``ECGDATAKIT_CACHE`` is set,
    otherwise ``ecgdatakit/hl7_aecg_schema`` in the user cache folder
    (``XDG_CACHE_HOME``, ``LOCALAPPDATA`` on Windows, or ``~/.cache``).
    """
    root = os.environ.get("ECGDATAKIT_CACHE")
    if root:
        return Path(root) / "hl7_aecg_schema"
    base = os.environ.get("XDG_CACHE_HOME") or os.environ.get("LOCALAPPDATA")
    return Path(base or Path.home() / ".cache") / "ecgdatakit" / "hl7_aecg_schema"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _missing_schema_files(directory: Path) -> list[str]:
    """Schema files that are absent from *directory* or differ from the
    known release."""
    return [name for name, digest in _SCHEMA_FILES.items()
            if not (directory / name).is_file() or _sha256(directory / name) != digest]


def fetch_hl7_aecg_schema(directory: str | Path | None = None, *, timeout: float = 30) -> Path:
    """Download the HL7 aECG schema and return its folder.

    The schema (``PORT_MT020001``, release 2003-12) is downloaded from the
    copy the FDA distributes with its aECG tools, at a fixed version. Every
    file must match a known SHA-256 hash, otherwise nothing is kept and
    :class:`~ecgdatakit.exceptions.ExportError` is raised. Files already
    present and matching are not downloaded again.

    Parameters
    ----------
    directory : str or Path, optional
        Where to keep the schema. Defaults to :func:`default_schema_dir`.
    timeout : float
        Timeout of each download, in seconds.
    """
    directory = Path(directory) if directory is not None else default_schema_dir()
    for name in _missing_schema_files(directory):
        url = _SCHEMA_URL + name
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                content = response.read()
        except OSError as e:
            raise ExportError(f"Cannot download the HL7 aECG schema file {url}: {e}") from e
        if hashlib.sha256(content).hexdigest() != _SCHEMA_FILES[name]:
            raise ExportError(
                f"The HL7 aECG schema file {url} differs from the expected release "
                "(SHA-256 mismatch); it was not kept")
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".part")
        partial.write_bytes(content)
        partial.replace(target)
    return directory


@functools.cache
def _load_schema(directory: Path):
    from lxml import etree

    return etree.XMLSchema(etree.parse(str(directory / _SCHEMA_MAIN)))


def _schema(directory: str | Path | None):
    try:
        import lxml  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Checking HL7 aECG files against the schema needs lxml: "
            "pip install 'ecgdatakit[export]'"
        ) from e
    directory = Path(directory) if directory is not None else default_schema_dir()
    missing = _missing_schema_files(directory)
    if missing:
        raise FileNotFoundError(
            f"The HL7 aECG schema is not in {directory} (missing or changed: {missing}). "
            "Download it with ecgdatakit.exporter.fetch_hl7_aecg_schema()"
        )
    return _load_schema(directory.resolve())


def validate_hl7_aecg(source: bytes | str | Path,
                      schema_dir: str | Path | None = None) -> list[str]:
    """Check an HL7 aECG document against the HL7 schema (``PORT_MT020001``).

    *source* is the XML content (``bytes``) or the path of a file. Returns
    the schema errors, one message per error, or an empty list when the
    document is valid.

    The schema is read from *schema_dir*, by default
    :func:`default_schema_dir`. Download it first with
    :func:`fetch_hl7_aecg_schema`. Needs lxml
    (``pip install 'ecgdatakit[export]'``).

    Raises
    ------
    FileNotFoundError
        When the schema is not in *schema_dir*, or differs from the known
        release.
    ImportError
        When lxml is not installed.
    """
    schema = _schema(schema_dir)
    from lxml import etree

    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True)
    try:
        if isinstance(source, bytes):
            doc = etree.fromstring(source, parser).getroottree()
        else:
            doc = etree.parse(str(source), parser)
    except etree.XMLSyntaxError as e:
        return [f"not well-formed XML: {e}"]
    if schema.validate(doc):
        return []
    return [f"line {err.line}: {err.message}" for err in schema.error_log]


class HL7aECGExporter(Exporter):
    """Write records as HL7 Annotated ECG (aECG) XML.

    Parameters
    ----------
    id_root : str
        OID or UUID of the organisation that assigned the patient, trial and
        device IDs. Defaults to :data:`DEFAULT_ID_ROOT`, a fixed UUID for IDs
        exported by ECGDataKit.
    trial_id : str, optional
        Clinical trial or study ID. The schema requires one: without it the
        trial ID is written with ``nullFlavor="NI"`` (no information).
    validate : bool
        Check every document against the HL7 schema before returning it, and
        raise :class:`~ecgdatakit.exceptions.ExportError` when it does not
        conform. Off by default: the schema is not shipped with ECGDataKit,
        download it once with :func:`fetch_hl7_aecg_schema`. Needs lxml
        (``pip install 'ecgdatakit[export]'``).
    schema_dir : str or Path, optional
        Folder of the HL7 schema used with *validate*. Defaults to
        :func:`default_schema_dir`.
    skip_unknown_leads : bool
        Leave out leads that have no MDC lead code (``CR1``...) with a
        warning. By default such a lead raises ``ExportError``.
    step_uv : float, optional
        Write every lead rounded to this step in µV instead of choosing the
        step from the ADC resolution.

    Raises
    ------
    ExportError
        From :meth:`to_bytes` or :meth:`export` when a lead has no voltage
        unit, NaN samples or no MDC code, or when the document does not
        conform to the schema.
    """

    FORMAT_NAME = "HL7 aECG"
    FILE_EXTENSION = ".xml"

    def __init__(
        self,
        *,
        id_root: str = DEFAULT_ID_ROOT,
        trial_id: str | None = None,
        validate: bool = False,
        schema_dir: str | Path | None = None,
        skip_unknown_leads: bool = False,
        step_uv: float | None = None,
    ) -> None:
        if step_uv is not None and step_uv <= 0:
            raise ValueError("step_uv must be positive")
        self.id_root = id_root
        self.trial_id = trial_id
        self.validate = validate
        self.schema_dir = schema_dir
        self.skip_unknown_leads = skip_unknown_leads
        self.step_uv = step_uv

    # -- document --------------------------------------------------------

    def to_bytes(self, record: ECGRecord) -> bytes:
        """The HL7 aECG document for *record*, as UTF-8 XML."""
        leads = self._coded(record.leads, "lead")
        if not leads:
            raise ExportError("The record has no lead to export")
        beats = self._coded(record.representative_beats, "representative beat")
        for _, lead in leads + beats:
            if lead.sampling_rate <= 0:
                raise ExportError(f"Lead '{lead.label}' has no sampling rate")

        start = record.recording.date
        duration = record.recording.duration or timedelta(
            seconds=len(leads[0][1].samples) / leads[0][1].sampling_rate)

        root = ET.Element(_tag("AnnotatedECG"), {"classCode": "OBS", "moodCode": "EVN"})
        _el(root, "id", root=str(uuid.uuid4()))
        _el(root, "code", code="93000", codeSystem=_CPT4, codeSystemName="CPT-4")
        self._effective_time(root, start, duration)
        self._trial(root, record)
        location = record.recording.location
        if location:
            site = _el(_el(root, "location"), "testingSite")
            _el(_el(site, "location"), "name", location)
        if record.patient.age is not None:
            obs = _el(_el(root, "controlVariable"), "relatedObservation")
            _el(obs, "code", code="21612-7", codeSystem=_LOINC, codeSystemName="LOINC",
                displayName="Reported Age")
            _el(obs, "value", xsi_type="PQ", value=str(record.patient.age), unit="a")
        if record.patient.clinical_history:
            finding = _el(_el(root, "pertainsTo"), "subjectFindingComment")
            _el(finding, "text", record.patient.clinical_history)

        rhythm = _el(_el(root, "component"), "series")
        self._series(rhythm, record, leads, "RHYTHM", "Rhythm Waveforms", start, duration)
        self._annotations(rhythm, record)
        if beats:
            derived = _el(_el(rhythm, "derivation"), "derivedSeries")
            self._series(derived, record, beats, "REPRESENTATIVE_BEAT",
                         "Representative Beat Waveforms", start, duration)
            self._wave_boundaries(derived, record)

        content = ET.tostring(root, encoding="UTF-8", xml_declaration=True)
        if self.validate:
            errors = validate_hl7_aecg(content, self.schema_dir)
            if errors:
                shown = "\n".join(errors[:10])
                raise ExportError(
                    f"The HL7 aECG document does not conform to the schema "
                    f"({len(errors)} errors):\n{shown}"
                )
        return content

    def _coded(self, leads: list[Lead], kind: str) -> list[tuple[str, Lead]]:
        """``(MDC code, lead)`` for every lead that has an MDC code."""
        coded, unknown = [], []
        for lead in leads:
            code = _lead_code(lead.label)
            if code is None:
                unknown.append(lead.label)
            else:
                coded.append((code, lead))
        if unknown:
            if not self.skip_unknown_leads:
                raise ExportError(
                    f"No MDC lead code for the {kind} labels {unknown}: HL7 aECG names "
                    "leads with MDC codes. Use skip_unknown_leads=True to leave them out"
                    + (" (XLE chest leads CR1 to CR6 become V1 to V6 with "
                       "XLEParser(wilson_correction=True))"
                       if any(label.startswith("CR") for label in unknown) else "")
                )
            warnings.warn(f"HL7 aECG export: {kind}s {unknown} have no MDC lead code "
                          "and were left out", UserWarning, stacklevel=3)
        return coded

    def _id(self, parent: ET.Element, extension: str | None, name: str = "id") -> None:
        if extension:
            _el(parent, name, root=self.id_root, extension=extension)
        else:
            _el(parent, name, nullFlavor="NI")

    @staticmethod
    def _effective_time(parent: ET.Element, start: datetime | None,
                        duration: timedelta) -> None:
        if start is None:
            _el(parent, "effectiveTime", nullFlavor="NI")
            return
        time = _el(parent, "effectiveTime")
        _el(time, "low", value=_ts(start), inclusive="true")
        _el(time, "high", value=_ts(start + duration), inclusive="false")

    def _trial(self, root: ET.Element, record: ECGRecord) -> None:
        """Subject and clinical trial, required by the schema."""
        event = _el(_el(root, "componentOf"), "timepointEvent")
        assignment = _el(_el(event, "componentOf"), "subjectAssignment")
        subject = _el(_el(assignment, "subject"), "trialSubject")
        patient = record.patient
        self._id(subject, patient.patient_id)
        sex = {"M": "M", "F": "F", "U": "UN"}.get(patient.sex.upper()) if patient.sex else None
        if patient.first_name or patient.last_name or sex or patient.birth_date:
            person = _el(subject, "subjectDemographicPerson")
            if patient.first_name or patient.last_name:
                name = _el(person, "name")
                if patient.first_name:
                    _el(name, "given", patient.first_name)
                if patient.last_name:
                    _el(name, "family", patient.last_name)
            if sex:
                _el(person, "administrativeGenderCode", code=sex, codeSystem=_GENDER,
                    codeSystemName="AdministrativeGender")
            if patient.birth_date:
                _el(person, "birthTime", value=patient.birth_date.strftime("%Y%m%d"))
        trial = _el(_el(assignment, "componentOf"), "clinicalTrial")
        self._id(trial, self.trial_id)
        institution = record.recording.device.institution
        if institution:
            site = _el(_el(trial, "location"), "trialSite")
            self._id(site, None)
            _el(_el(site, "location"), "name", institution)

    # -- series ----------------------------------------------------------

    def _series(self, series: ET.Element, record: ECGRecord, leads: list[tuple[str, Lead]],
                code: str, display: str, start: datetime | None, duration: timedelta) -> None:
        _el(series, "id", root=str(uuid.uuid4()))
        _el(series, "code", code=code, codeSystem=_ACT_CODE, codeSystemName="ActCode",
            displayName=display)
        self._effective_time(series, start, duration)
        self._device(series, record)
        if record.recording.technician:
            performer = _el(_el(series, "secondaryPerformer"), "seriesPerformer")
            _el(_el(performer, "assignedPerson"), "name", record.recording.technician)
        if code == "RHYTHM":
            self._filters(series, record)

        groups: dict[tuple[int, int], list[tuple[str, Lead]]] = {}
        for item in leads:
            lead = item[1]
            groups.setdefault((lead.sampling_rate, len(lead.samples)), []).append(item)
        absolute = code == "RHYTHM" and start is not None
        rounded: list[tuple[str, float, float]] = []
        for (rate, _), group in groups.items():
            sequence_set = _el(_el(series, "component"), "sequenceSet")
            self._time_sequence(sequence_set, rate, start if absolute else None)
            for lead_code, lead in group:
                step, error = self._lead_sequence(sequence_set, lead_code, lead)
                if error:
                    rounded.append((lead.label, step, error))
        if rounded:
            kind = "rhythm leads" if code == "RHYTHM" else "representative beats"
            steps = sorted({f"{step:.6g}" for _, step, _ in rounded})
            warnings.warn(
                f"HL7 aECG export: {kind} {[label for label, _, _ in rounded]} are not a "
                f"whole number of ADC counts and were rounded to {' or '.join(steps)} uV "
                f"(largest error {max(e for _, _, e in rounded):.3g} uV)",
                UserWarning, stacklevel=3)

    def _device(self, series: ET.Element, record: ECGRecord) -> None:
        device = record.recording.device
        if not (device.manufacturer or device.model or device.software_version
                or device.serial_number):
            return
        author = _el(_el(series, "author"), "seriesAuthor")
        series_device = _el(author, "manufacturedSeriesDevice")
        if device.serial_number:
            _el(series_device, "id", root=self.id_root, extension=device.serial_number)
        if device.model:
            _el(series_device, "manufacturerModelName", device.model)
        if device.software_version:
            _el(series_device, "softwareName", device.software_version)
        if device.manufacturer:
            _el(_el(author, "manufacturerOrganization"), "name", device.manufacturer)

    @staticmethod
    def _filters(series: ET.Element, record: ECGRecord) -> None:
        filters = record.recording.acquisition.filters
        for kind, value, part in (
            ("LOW_PASS", filters.lowpass, "CUTOFF_FREQ"),
            ("HIGH_PASS", filters.highpass, "CUTOFF_FREQ"),
            ("NOTCH", filters.notch, "NOTCH_FREQ"),
        ):
            if value is None or value <= 0:
                continue
            variable = _el(_el(series, "controlVariable"), "controlVariable")
            _el(variable, "code", code=f"MDC_ECG_CTL_VBL_ATTR_FILTER_{kind}",
                codeSystem=_MDC, codeSystemName="MDC")
            sub = _el(_el(variable, "component"), "controlVariable")
            _el(sub, "code", code=f"MDC_ECG_CTL_VBL_ATTR_FILTER_{part}", codeSystem=_MDC,
                codeSystemName="MDC")
            _el(sub, "value", xsi_type="PQ", value=_number(value), unit="Hz")

    @staticmethod
    def _time_sequence(sequence_set: ET.Element, rate: int, start: datetime | None) -> None:
        sequence = _el(_el(sequence_set, "component"), "sequence")
        increment = _number(1.0 / rate)
        if start is not None:
            _el(sequence, "code", code="TIME_ABSOLUTE", codeSystem=_ACT_CODE,
                codeSystemName="ActCode")
            value = _el(sequence, "value", xsi_type="GLIST_TS")
            _el(value, "head", value=_ts(start))
        else:
            _el(sequence, "code", code="TIME_RELATIVE", codeSystem=_ACT_CODE,
                codeSystemName="ActCode")
            value = _el(sequence, "value", xsi_type="GLIST_PQ")
            _el(value, "head", value="0", unit="s")
        _el(value, "increment", value=increment, unit="s")

    def _lead_sequence(self, sequence_set: ET.Element, code: str,
                       lead: Lead) -> tuple[float, float]:
        """Write the sequence of *lead*, return its scale and rounding error (µV)."""
        digits, scale, origin, error = _digits(lead, self.step_uv)
        sequence = _el(_el(sequence_set, "component"), "sequence")
        _el(sequence, "code", code=code, codeSystem=_MDC, codeSystemName="MDC")
        value = _el(sequence, "value", xsi_type="SLIST_PQ")
        _el(value, "origin", value=_number(origin), unit="uV")
        _el(value, "scale", value=_number(scale), unit="uV")
        _el(value, "digits", " ".join(map(str, digits.tolist())))
        return scale, error

    # -- annotations -----------------------------------------------------

    def _annotations(self, rhythm: ET.Element, record: ECGRecord) -> None:
        """Global measurements, interpretation and rhythm statement."""
        annotations: list[tuple[str, dict, list]] = []
        measurements = record.measurements
        for field, code, unit in _MEASUREMENTS:
            value = getattr(measurements, field)
            if value is not None:
                annotations.append((code, {"xsi_type": "PQ", "value": _number(value),
                                           "unit": unit}, []))
        interpretation = record.interpretation
        statements = [text for text, _ in interpretation.statements if text]
        if statements:
            annotations.append(("MDC_ECG_INTERPRETATION", {}, [
                ("MDC_ECG_INTERPRETATION_STATEMENT", text) for text in statements]))
        rhythm_text = record.annotations.get("rhythm")
        if rhythm_text:
            annotations.append(("MDC_ECG_RHY", {"xsi_type": "ST", "text": rhythm_text}, []))
        if not annotations:
            return

        annotation_set = _el(_el(rhythm, "subjectOf"), "annotationSet")
        if interpretation.interpretation_date:
            _el(annotation_set, "activityTime", value=_ts(interpretation.interpretation_date))
        if interpretation.source in ("overread", "confirmed") and interpretation.interpreter:
            author_type = _el(_el(_el(annotation_set, "author"), "assignedEntity"),
                              "assignedAuthorType")
            _el(_el(author_type, "assignedPerson"), "name", interpretation.interpreter)
        for code, value, children in annotations:
            annotation = _el(_el(annotation_set, "component"), "annotation")
            _el(annotation, "code", code=code, codeSystem=_MDC, codeSystemName="MDC")
            if value:
                text = value.pop("text", None)
                _el(annotation, "value", text, **value)
            for child_code, text in children:
                child = _el(_el(annotation, "component"), "annotation")
                _el(child, "code", code=child_code, codeSystem=_MDC, codeSystemName="MDC")
                _el(child, "value", text, xsi_type="ST")

    @staticmethod
    def _wave_boundaries(derived: ET.Element, record: ECGRecord) -> None:
        """P, QRS and T onsets and offsets of the representative beat, in ms
        from its start."""
        waves = []
        for prefix, code in _WAVES:
            edges = {}
            for edge, key in (("low", "onset"), ("high", "offset")):
                text = record.annotations.get(f"{prefix}_{key}_ms")
                try:
                    edges[edge] = float(text) if text is not None else None
                except ValueError:
                    edges[edge] = None
            if any(v is not None for v in edges.values()):
                waves.append((code, edges))
        if not waves:
            return
        annotation_set = _el(_el(derived, "subjectOf"), "annotationSet")
        for code, edges in waves:
            annotation = _el(_el(annotation_set, "component"), "annotation")
            _el(annotation, "code", code="MDC_ECG_WAVC", codeSystem=_MDC, codeSystemName="MDC")
            _el(annotation, "value", xsi_type="CE", code=code, codeSystem=_MDC,
                codeSystemName="MDC")
            roi = _el(_el(annotation, "support"), "supportingROI", classCode="ROIBND")
            _el(roi, "code", code="ROIPS", codeSystem=_ACT_CODE, codeSystemName="ActCode")
            boundary = _el(_el(roi, "component"), "boundary")
            _el(boundary, "code", code="TIME_RELATIVE", codeSystem=_ACT_CODE,
                codeSystemName="ActCode")
            interval = _el(boundary, "value", xsi_type="IVL_PQ")
            for edge, value in edges.items():
                if value is not None:
                    _el(interval, edge, value=_number(value), unit="ms")

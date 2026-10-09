"""HL7 Annotated ECG (aECG) XML format parser.

Follows ANSI/HL7 V3 ECG R1-2004 (PORT_MT020001) and the FDA aECG
implementation guide. Waveforms are SLIST_PQ sequences where
``value = origin + scale * digit``; the time base is a GLIST
(``TIME_ABSOLUTE`` or ``TIME_RELATIVE``) whose ``increment`` gives the
sampling interval.
"""

from __future__ import annotations

import math
import re
import warnings
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ecgdatakit.exceptions import CorruptedFileError
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
    _TO_UV,
    _normalize_unit,
    derive_is_raw,
)
from ecgdatakit.parsing.helpers import (
    fill_signal_summary,
    normalize_lead_label,
    unique_labels,
)
from ecgdatakit.parsing.helpers.xml import header_text, parse_xml_root
from ecgdatakit.parsing.parser import Parser

# Lead codes whose label is not the code suffix
_LEAD_CODE_LABELS = {
    "AVRNEG": "-aVR",
}

_MEASUREMENT_MAP: dict[str, tuple[str, str]] = {
    # code: (GlobalMeasurements field, quantity)
    "MDC_ECG_HEART_RATE": ("heart_rate", "rate"),
    "MDC_ECG_TIME_PD_RR": ("rr_interval", "time"),
    "MDC_ECG_TIME_PD_RR_GL": ("rr_interval", "time"),
    "MDC_ECG_TIME_PD_PR": ("pr_interval", "time"),
    "MDC_ECG_TIME_PD_PR_GL": ("pr_interval", "time"),
    "MDC_ECG_TIME_PD_QRS": ("qrs_duration", "time"),
    "MDC_ECG_TIME_PD_QRS_GL": ("qrs_duration", "time"),
    "MDC_ECG_TIME_PD_QT": ("qt_interval", "time"),
    "MDC_ECG_TIME_PD_QT_GL": ("qt_interval", "time"),
    "MDC_ECG_TIME_PD_QTcB": ("qtc_bazett", "time"),
    "MDC_ECG_TIME_PD_QTcF": ("qtc_fridericia", "time"),
    "MDC_ECG_ANGLE_P_FRONT": ("p_axis", "angle"),
    "MDC_ECG_ANGLE_P_FRONT_GL": ("p_axis", "angle"),
    "MDC_ECG_ANGLE_QRS_FRONT": ("qrs_axis", "angle"),
    "MDC_ECG_ANGLE_QRS_FRONT_GL": ("qrs_axis", "angle"),
    "MDC_ECG_ANGLE_T_FRONT": ("t_axis", "angle"),
    "MDC_ECG_ANGLE_T_FRONT_GL": ("t_axis", "angle"),
    "MDC_ECG_QRS_COUNT": ("qrs_count", "count"),
    "MINDRAY_ECG_P_AXIS": ("p_axis", "angle"),
    "MINDRAY_ECG_QRS_AXIS": ("qrs_axis", "angle"),
    "MINDRAY_ECG_T_AXIS": ("t_axis", "angle"),
}

_TIME_TO_MS = {"ms": 1.0, "s": 1000.0, "us": 0.001, "µs": 0.001, "min": 60000.0}
_RATE_TO_BPM = {"bpm": 1.0, "/min": 1.0, "1/min": 1.0, "{beats}/min": 1.0,
                "{beat}/min": 1.0, "Hz": 60.0, "/s": 60.0}
_ANGLE_TO_DEG = {"deg": 1.0, "rad": 180.0 / math.pi}
_INCREMENT_TO_S = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6}
_AGE_TO_YEARS = {"a": 1.0, "yr": 1.0, "mo": 1 / 12, "wk": 7 / 365.25, "d": 1 / 365.25,
                 "h": 1 / 8766}

_INTERP_STATEMENT_CODES = (
    "MDC_ECG_INTERPRETATION",
    "MDC_ECG_INTERPRETATION_STATEMENT",
    "MDC_ECG_INTERPRETATION_SUMMARY",
)
_SEVERITY_CODES = ("MDC_ECG_INTERPRETATION_SEVERITY",)
_COMMENT_CODES = ("MDC_ECG_INTERPRETATION_COMMENT",)

_TS_RE = re.compile(
    r"^(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?(?:\.(\d+))?([+-]\d{4})?$"
)
_CREATED_RE = re.compile(rb"File created on (\d{2})/(\d{2})/(\d{4})")


def _parse_ts(value: str | None, need_time: bool = False) -> datetime | None:
    """Parse an HL7 TS value (``YYYYMMDDHHMMSS.UUUU[+|-ZZzz]``).

    Precision may stop at any component. Returns ``None`` when the value is
    malformed, when it has less than day precision, or (with *need_time*)
    when it has no minute, so no time of day is filled in. A time zone
    offset gives an aware datetime.
    """
    if not value:
        return None
    m = _TS_RE.match(value.strip())
    if not m:
        return None
    year, month, day, hour, minute, second, frac, tz = m.groups()
    if day is None or (need_time and minute is None):
        return None
    micro = int(frac[:6].ljust(6, "0")) if frac else 0
    tzinfo = None
    if tz:
        sign = 1 if tz[0] == "+" else -1
        tzinfo = timezone(sign * timedelta(hours=int(tz[1:3]), minutes=int(tz[3:5])))
    try:
        return datetime(
            int(year), int(month), int(day), int(hour or 0), int(minute or 0),
            int(second or 0), micro, tzinfo=tzinfo,
        )
    except ValueError:
        return None


def _float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _child(el: ET.Element | None, path: str) -> ET.Element | None:
    return None if el is None else el.find(path)


def _children(el: ET.Element | None, path: str) -> list[ET.Element]:
    return [] if el is None else el.findall(path)


def _text(el: ET.Element | None) -> str:
    """Whitespace-normalised text content of *el* (including children)."""
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split())


def _attr(el: ET.Element | None, name: str) -> str:
    if el is None:
        return ""
    return (el.get(name) or "").strip()


def _placeholder(value: str) -> str:
    """Return ``""`` for placeholder identifiers such as ``"?"``."""
    return "" if value in ("?", "UNK", "NA", "N/A") else value


def _person_name(el: ET.Element | None) -> tuple[str, str, str]:
    """Return ``(family, given, full_text)`` of a PN ``<name>`` element.

    Multiple given or family parts are joined with spaces. ``full_text`` is
    the whole name when it is a plain string without parts.
    """
    if el is None:
        return "", "", ""
    family = " ".join(_text(p) for p in el.findall("family") if _text(p))
    given = " ".join(_text(p) for p in el.findall("given") if _text(p))
    if family or given:
        parts = [_text(p) for p in el.findall("prefix")] + [given, family]
        parts += [_text(p) for p in el.findall("suffix")]
        return family, given, " ".join(p for p in parts if p)
    return "", "", _text(el)


def _code_text(value: ET.Element | None) -> str:
    """Human-readable content of an annotation value (ST, CE or PQ)."""
    if value is None:
        return ""
    text = _text(value)
    if text:
        return text
    for attr in ("displayName", "code", "value"):
        if value.get(attr):
            unit = value.get("unit", "")
            v = value.get(attr).strip()
            return f"{v} {unit}".strip() if attr == "value" and unit not in ("", "1") else v
    return ""


class HL7aECGParser(Parser):
    """Parser for HL7 Annotated ECG (aECG) XML files."""

    FORMAT_NAME = "HL7 aECG"
    FORMAT_DESCRIPTION = "HL7 Annotated ECG XML format (FDA submission standard)"
    FILE_EXTENSIONS = [".xml"]
    PRIORITY = 50

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        text = header_text(header)
        return re.search(r"<([A-Za-z_][\w.-]*:)?AnnotatedECG\b", text) is not None

    def parse(self, file_path: Path) -> ECGRecord:
        self._warnings: list[str] = []
        raw = Path(file_path).read_bytes()
        try:
            root = parse_xml_root(raw)
        except ET.ParseError as e:
            raise CorruptedFileError(f"Malformed XML in {file_path}: {e}") from e
        if root.tag != "AnnotatedECG":
            raise CorruptedFileError(f"Missing AnnotatedECG root element in {file_path}")

        record = ECGRecord(source_format="hl7_aecg")
        meta = record.raw_metadata
        meta["filepath"] = str(file_path)
        meta["document_id"] = _attr(_child(root, "id"), "root")

        rhythm, derived = self._select_series(root, file_path)
        record.patient = self._read_patient(root)
        record.recording = self._read_recording(root, rhythm, meta)

        record.leads, rhythm_meta = self._read_sequence_set(rhythm, "rhythm", file_path)
        if not record.leads:
            raise CorruptedFileError(f"No ECG lead sequence in the rhythm series of {file_path}")
        meta["rhythm_time"] = rhythm_meta
        if derived is not None:
            record.median_beats, beat_meta = self._read_sequence_set(derived, "median", file_path)
            meta["median_time"] = beat_meta

        self._complete_dates(record, rhythm_meta)
        self._read_device(rhythm, record.recording)
        record.recording.acquisition.filters = self._read_filters(rhythm)
        self._read_annotations(record, rhythm, derived)
        self._read_context(root, record)

        lead = record.leads[0]
        record.recording.acquisition.signal = SignalCharacteristics(
            sampling_rate=lead.sampling_rate,
            data_encoding="decimal",
            compression="none",
            number_channels_valid=len(record.leads),
            number_channels_allocated=len(record.leads),
        )
        if len({ld.sampling_rate for ld in record.leads}) > 1:
            record.recording.acquisition.signal.sampling_rate = 0
        fill_signal_summary(record)

        if record.patient.age is None and record.patient.birth_date and record.recording.date:
            acq = record.recording.date.date()
            dob = record.patient.birth_date.date()
            record.patient.age = acq.year - dob.year - ((acq.month, acq.day) < (dob.month, dob.day))

        record.file_format = FileFormatInfo(
            version=_attr(root, "ITSVersion"),
            creation_date=self._creation_date(raw),
        )

        for message in self._warnings:
            warnings.warn(message, stacklevel=2)
        return record

    # ------------------------------------------------------------------
    # Series and waveforms
    # ------------------------------------------------------------------

    def _select_series(
        self, root: ET.Element, file_path: Path
    ) -> tuple[ET.Element, ET.Element | None]:
        """Return the rhythm series and the representative beat series."""
        rhythm: list[ET.Element] = []
        beats: list[ET.Element] = []
        for series in _children(root, "component/series"):
            code = _attr(_child(series, "code"), "code").upper()
            if code == "REPRESENTATIVE_BEAT":
                beats.append(series)
            else:
                rhythm.append(series)
            for derived in _children(series, "derivation/derivedSeries"):
                if _attr(_child(derived, "code"), "code").upper() in ("REPRESENTATIVE_BEAT", ""):
                    beats.append(derived)
        if not rhythm:
            raise CorruptedFileError(f"No rhythm series in {file_path}")
        if len(rhythm) > 1:
            self._warnings.append(
                f"{len(rhythm)} rhythm series found; only the first is read"
            )
        if len(beats) > 1:
            self._warnings.append(
                f"{len(beats)} representative beat series found; only the first is read"
            )
        return rhythm[0], (beats[0] if beats else None)

    def _read_sequence_set(
        self, series: ET.Element, kind: str, file_path: Path
    ) -> tuple[list[Lead], dict]:
        sets = _children(series, "component/sequenceSet")
        if not sets:
            return [], {}
        if len(sets) > 1:
            self._warnings.append(
                f"{len(sets)} sequence sets in the {kind} series; only the first is read"
            )

        time_info: dict = {}
        entries: list[tuple[str, np.ndarray, ET.Element]] = []
        skipped: list[str] = []
        for seq in _children(sets[0], "component/sequence"):
            code = _attr(_child(seq, "code"), "code")
            value = _child(seq, "value")
            if code.upper().startswith("TIME_"):
                if not time_info:
                    time_info = self._read_time(code, value)
                continue
            label = self._lead_label(code)
            if label is None:
                skipped.append(code)
                continue
            digits = _child(value, "digits")
            if digits is None or not (digits.text or "").split():
                raise CorruptedFileError(f"Lead {code} has no digits in {file_path}")
            try:
                samples = np.array(digits.text.split(), dtype=np.float64)
            except ValueError as e:
                raise CorruptedFileError(f"Invalid digits in lead {code} of {file_path}: {e}") from e
            entries.append((label, samples, value))
        if skipped:
            self._warnings.append(f"Unrecognised sequences ignored in the {kind} series: {skipped}")
        if entries and not time_info.get("sampling_rate"):
            raise CorruptedFileError(f"No usable time increment in the {kind} series of {file_path}")

        labels = unique_labels([label for label, _, _ in entries])
        leads = []
        for label, (_, samples, value) in zip(labels, entries):
            leads.append(self._make_lead(label, samples, value, time_info["sampling_rate"]))
        return leads, time_info

    def _read_time(self, code: str, value: ET.Element | None) -> dict:
        head = _child(value, "head")
        inc = _child(value, "increment")
        info = {
            "code": code,
            "head": _attr(head, "value"),
            "head_unit": _attr(head, "unit"),
            "increment": _attr(inc, "value"),
            "increment_unit": _attr(inc, "unit"),
            "sampling_rate": 0,
        }
        step = _float(info["increment"])
        if not step or step <= 0:
            return info
        unit = info["increment_unit"]
        if unit not in _INCREMENT_TO_S:
            raise CorruptedFileError(
                f"HL7 aECG: time increment unit {unit!r} is missing or not supported, "
                "the sampling rate cannot be determined"
            )
        seconds = step * _INCREMENT_TO_S[unit]
        rate = 1.0 / seconds
        info["sampling_rate_exact"] = rate
        info["sampling_rate"] = int(round(rate))
        if abs(rate - round(rate)) > 1e-3:
            self._warnings.append(
                f"Sampling rate {rate:.4f} Hz is not an integer; rounded to {round(rate)} Hz"
            )
        return info

    @staticmethod
    def _lead_label(code: str) -> str | None:
        """Map an MDC (or vendor) lead code to a standard label."""
        if "_ECG_LEAD_" not in code.upper():
            return None
        suffix = code[code.upper().index("_ECG_LEAD_") + len("_ECG_LEAD_"):]
        if not suffix:
            return None
        return _LEAD_CODE_LABELS.get(suffix.upper()) or normalize_lead_label(suffix)

    @staticmethod
    def _make_lead(label: str, samples: np.ndarray, value: ET.Element, rate: int) -> Lead:
        scale_el = _child(value, "scale")
        origin_el = _child(value, "origin")
        scale = _float(_attr(scale_el, "value"))
        origin = _float(_attr(origin_el, "value")) or 0.0
        scale_unit_raw = _attr(scale_el, "unit")
        origin_unit = _normalize_unit(_attr(origin_el, "unit")) if _attr(origin_el, "unit") else None
        unit = _normalize_unit(scale_unit_raw) if scale_unit_raw else None

        resolution = scale if scale is not None else 1.0
        offset = origin
        if unit and origin_unit and origin_unit != unit:
            offset = origin * _TO_UV[origin_unit] / _TO_UV[unit]
        res_unit = unit or ""
        raw = derive_is_raw(resolution, offset, res_unit)
        return Lead(
            label=label,
            samples=samples,
            sampling_rate=rate,
            resolution=resolution,
            resolution_unit=res_unit,
            offset=offset,
            units="" if raw else res_unit,
            is_raw=raw,
            adc_resolution=scale if scale is not None else 0.0,
            adc_resolution_unit=scale_unit_raw,
        )

    # ------------------------------------------------------------------
    # Header sections
    # ------------------------------------------------------------------

    def _read_patient(self, root: ET.Element) -> PatientInfo:
        info = PatientInfo()
        subject = _child(root, "componentOf/timepointEvent/componentOf/subjectAssignment/subject/trialSubject")
        if subject is None:
            return info
        info.patient_id = _placeholder(_attr(_child(subject, "id"), "extension"))
        person = _child(subject, "subjectDemographicPerson")
        if person is None:
            return info

        family, given, full = _person_name(_child(person, "name"))
        if family or given:
            info.last_name, info.first_name = family, given
        elif full:
            # Plain-text name (usually the subject's initials)
            info.last_name = full

        gender = _attr(_child(person, "administrativeGenderCode"), "code").upper()
        if gender:
            info.sex = gender if gender in ("M", "F") else "U"
        info.birth_date = _parse_ts(_attr(_child(person, "birthTime"), "value"))
        race = _child(person, "raceCode")
        info.race = _attr(race, "displayName") or _attr(race, "code")
        return info

    def _read_recording(self, root: ET.Element, rhythm: ET.Element, meta: dict) -> RecordingInfo:
        info = RecordingInfo()
        for node in (_child(root, "effectiveTime"), _child(rhythm, "effectiveTime")):
            if node is None:
                continue
            low = _attr(_child(node, "low"), "value") or _attr(_child(node, "center"), "value")
            high = _attr(_child(node, "high"), "value")
            if low and info.date is None:
                info.date = _parse_ts(low, need_time=True)
                meta.setdefault("effective_time", low)
            if high and info.end_date is None:
                info.end_date = _parse_ts(high, need_time=True)
            if info.date:
                break
        return info

    def _complete_dates(self, record: ECGRecord, time_info: dict) -> None:
        rec = record.recording
        if rec.date is None and time_info.get("code", "").upper() == "TIME_ABSOLUTE":
            rec.date = _parse_ts(time_info.get("head"), need_time=True)
        same_kind = rec.date and rec.end_date and (
            (rec.date.tzinfo is None) == (rec.end_date.tzinfo is None))
        if same_kind and rec.end_date >= rec.date:
            rec.duration = rec.end_date - rec.date
        elif record.leads and record.leads[0].sampling_rate:
            lead = record.leads[0]
            rec.duration = timedelta(seconds=len(lead.samples) / time_info["sampling_rate_exact"])

    def _read_device(self, series: ET.Element, rec: RecordingInfo) -> None:
        dev: DeviceInfo = rec.device
        author = _child(series, "author/seriesAuthor")
        device = _child(author, "manufacturedSeriesDevice")
        if device is not None:
            dev.model = _text(_child(device, "manufacturerModelName"))
            dev.software_version = _text(_child(device, "softwareName"))
            serial = _text(_child(device, "SerialNumber")) or _text(_child(device, "serialNumber"))
            if not serial:
                serial = _placeholder(_attr(_child(device, "id"), "extension"))
                serial = "" if serial == "0" else serial
            dev.serial_number = serial
            code = _child(device, "code")
            dev.acquisition_type = _attr(code, "displayName") or _attr(code, "code")
        dev.manufacturer = _text(_child(author, "manufacturerOrganization/name"))

        for path in ("secondaryPerformer", "performer"):
            for perf in _children(series, path):
                _, _, name = _person_name(_child(perf, "seriesPerformer/assignedPerson/name"))
                ident = _attr(_child(perf, "seriesPerformer/id"), "extension")
                if name or ident:
                    rec.technician = name or ident
                    return

    @staticmethod
    def _read_filters(series: ET.Element) -> FilterSettings:
        fs = FilterSettings()
        for cv in _children(series, "controlVariable/controlVariable"):
            kind = _attr(_child(cv, "code"), "code").upper()
            freq = None
            for sub in _children(cv, "component/controlVariable"):
                sub_code = _attr(_child(sub, "code"), "code").upper()
                value = _child(sub, "value")
                if sub_code.endswith(("_CUTOFF_FREQ", "_NOTCH_FREQ")):
                    v = _float(_attr(value, "value"))
                    unit = _attr(value, "unit") or "Hz"
                    if v is not None and unit in ("Hz", "kHz"):
                        freq = v * (1000.0 if unit == "kHz" else 1.0)
            if freq is not None and freq <= 0:
                freq = None
            if kind.endswith("_FILTER_LOW_PASS"):
                fs.lowpass = freq
            elif kind.endswith("_FILTER_HIGH_PASS"):
                fs.highpass = freq
            elif kind.endswith("_FILTER_NOTCH"):
                fs.notch = freq
                fs.notch_active = True
        return fs

    def _read_context(self, root: ET.Element, record: ECGRecord) -> None:
        """Trial, site, investigator and other context elements."""
        meta = record.raw_metadata
        rec = record.recording
        event = _child(root, "componentOf/timepointEvent")
        trial = _child(event, "componentOf/subjectAssignment/componentOf/clinicalTrial")

        site_loc = _child(trial, "location/trialSite/location")
        rec.device.institution = _text(_child(site_loc, "name"))
        testing = _child(root, "location/testingSite/location")
        for loc in (testing, site_loc):
            if loc is None:
                continue
            addr = _child(loc, "addr")
            parts = [_text(_child(loc, "name"))] if loc is testing else []
            parts += [_text(_child(addr, p)) for p in ("city", "state", "country")]
            text = ", ".join(p for p in parts if p)
            if text:
                rec.location = text
                break

        def code_of(el: ET.Element | None) -> str:
            return _attr(el, "displayName") or _attr(el, "code")

        context = {
            "reason": code_of(_child(root, "reasonCode")),
            "confidentiality": code_of(_child(root, "confidentialityCode")),
            "visit": code_of(_child(event, "code")),
            "visit_reason": code_of(_child(event, "reasonCode")),
            "trial_id": _attr(_child(trial, "id"), "extension"),
            "trial_title": _text(_child(trial, "title")),
            "protocol_id": _attr(_child(trial, "componentOf/clinicalTrialProtocol/id"), "extension"),
            "sponsor": _text(_child(trial, "author/clinicalTrialSponsor/sponsorOrganization/name")),
            "trial_site_id": _attr(_child(trial, "location/trialSite/id"), "extension"),
            "investigator": _person_name(_child(
                trial, "responsibleParty/trialInvestigator/investigatorPerson/name"))[2],
            "treatment_group": code_of(_child(
                event, "componentOf/subjectAssignment/definition/treatmentGroupAssignment/code")),
            "timepoint": code_of(_child(root, "definition/relativeTimepoint/code")),
            "study_event_performer": _person_name(_child(
                event, "performer/studyEventPerformer/assignedPerson/name"))[2],
        }
        for key, value in context.items():
            if _placeholder(value):
                meta[key] = value

        # Subject findings (free text) and related observations such as age
        notes = [_text(_child(p, "subjectFindingComment/text")) for p in _children(root, "pertainsTo")]
        notes = [n for n in notes if n]
        if notes:
            record.patient.clinical_history = "\n".join(notes)
        observations = {}
        for obs in _children(root, "controlVariable/relatedObservation"):
            code = _child(obs, "code")
            value = _child(obs, "value")
            key = _attr(code, "displayName") or _attr(code, "code")
            if _attr(code, "code") == "21612-7":  # LOINC: reported age
                age = _float(_attr(value, "value"))
                factor = _AGE_TO_YEARS.get(_attr(value, "unit") or "a")
                if age is not None and factor:
                    record.patient.age = int(age * factor)  # stated age wins
            if key:
                observations[key] = _code_text(value)
        if observations:
            meta["related_observations"] = observations

    @staticmethod
    def _creation_date(raw: bytes) -> date | None:
        """File creation date from the writer's header comment, when present."""
        m = _CREATED_RE.search(raw[:2048])
        if not m:
            return None
        try:
            return date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    def _read_annotations(
        self, record: ECGRecord, rhythm: ET.Element, derived: ET.Element | None
    ) -> None:
        machine = _AnnotationSink()
        expert = _AnnotationSink()
        for series, relative in ((rhythm, False), (derived, True)):
            for aset in _children(series, "subjectOf/annotationSet"):
                sink = self._classify_set(aset, machine, expert)
                for ann in _children(aset, "component/annotation"):
                    self._walk(ann, sink, record, relative, in_beat=False)

        # Measurements: expert values win over machine values
        gm = GlobalMeasurements()
        for sink in (expert, machine):
            for name, value in sink.measurements.items():
                if getattr(gm, name) is None:
                    setattr(gm, name, value)
        # Per-beat values and wave boundaries are reported as stored: no
        # median or interval is computed from them
        beats = {}
        for sink in (machine, expert):
            beats.update({name: list(v) for name, v in sink.beat_values.items() if v})
            beats.update({code: [f"{v:g} {u}".strip() for v, u in vals]
                          for code, vals in sink.beat_extra.items() if vals})
        if beats:
            record.raw_metadata["beat_measurements"] = beats
        self._report_fiducials(expert.fiducials or machine.fiducials, record)
        record.measurements = gm
        for sink in (machine, expert):
            record.annotations.update(sink.extra)

        interp = Interpretation()
        if expert.statements or expert.severity:
            interp.statements = expert.statements
            interp.severity = expert.severity
            interp.source = "confirmed" if expert.confirmed else "overread"
            interp.interpreter = expert.author
            interp.interpretation_date = expert.date
            if machine.statements:
                record.annotations["machine_interpretation"] = "\n".join(
                    s for s, _ in machine.statements
                )
        elif machine.statements or machine.severity:
            interp.statements = machine.statements
            interp.severity = machine.severity
            interp.source = "machine"
            interp.interpretation_date = machine.date
        record.interpretation = interp
        comments = machine.comments + expert.comments
        if comments:
            record.annotations["interpretation_comment"] = "\n".join(comments)
        lead_meas = {**machine.lead_measurements, **expert.lead_measurements}
        if lead_meas:
            record.raw_metadata["lead_measurements"] = lead_meas

    @staticmethod
    def _classify_set(
        aset: ET.Element, machine: _AnnotationSink, expert: _AnnotationSink
    ) -> _AnnotationSink:
        """Pick the sink for an annotation set from its author."""
        entity = _child(aset, "author/assignedEntity")
        person = _child(entity, "assignedAuthorType/assignedPerson")
        activity = _parse_ts(_attr(_child(aset, "activityTime"), "value"), need_time=True)
        if person is None:
            if activity and machine.date is None:
                machine.date = activity
            return machine
        if not expert.author:
            expert.author = _person_name(_child(person, "name"))[2] or _attr(
                _child(entity, "id"), "extension")
        if activity and expert.date is None:
            expert.date = activity
        if _child(aset, "legalAuthenticator") is not None or _child(
                entity, "legalAuthenticator") is not None:
            expert.confirmed = True
        return expert

    def _walk(
        self, ann: ET.Element, sink: _AnnotationSink, record: ECGRecord,
        relative: bool, in_beat: bool,
    ) -> None:
        code = _attr(_child(ann, "code"), "code")
        value = _child(ann, "value")
        lead_roi = self._lead_roi(ann)
        upper = code.upper()

        if upper == "MDC_ECG_BEAT":
            for sub in _children(ann, "component/annotation"):
                self._walk(sub, sink, record, relative, in_beat=True)
            return
        if lead_roi:
            label = self._lead_label(lead_roi) or lead_roi
            store = sink.lead_measurements.setdefault(label, {})
            for sub in [ann] + _children(ann, ".//annotation"):
                sub_code = _attr(_child(sub, "code"), "code")
                text = _code_text(_child(sub, "value"))
                if sub_code and text:
                    store[sub_code] = text
            return

        if upper in _SEVERITY_CODES:
            sink.severity = self._severity(value)
        elif upper in _COMMENT_CODES:
            text = _code_text(value)
            if text:
                sink.comments.append(text)
        elif upper in _INTERP_STATEMENT_CODES:
            text = _code_text(value)
            if text and (text, "") not in sink.statements:
                sink.statements.append((text, ""))
        elif upper == "MDC_ECG_RHY":
            text = _code_text(value)
            if text:
                sink.extra.setdefault("rhythm", text)
        elif upper in ("MDC_ECG_WAVC", "MDC_ECG_WAVC_TYPE"):
            if relative and not in_beat:
                self._fiducial(ann, value, sink)
        elif code in _MEASUREMENT_MAP and value is not None and _attr(value, "value"):
            field, quantity = _MEASUREMENT_MAP[code]
            converted = self._convert(_attr(value, "value"), _attr(value, "unit"), quantity, code)
            if converted is not None:
                if in_beat:
                    sink.beat_values.setdefault(field, []).append(converted)
                else:
                    sink.measurements.setdefault(field, int(round(converted)))
        elif value is not None and _float(_attr(value, "value")) is not None:
            # Measurement without a home in GlobalMeasurements
            if in_beat:
                sink.beat_extra.setdefault(code, []).append(
                    (_float(_attr(value, "value")), _attr(value, "unit")))
            else:
                sink.extra.setdefault(code, _code_text(value))

        for sub in _children(ann, "component/annotation"):
            self._walk(sub, sink, record, relative, in_beat)

    @staticmethod
    def _lead_roi(ann: ET.Element) -> str:
        """Lead code when the annotation is restricted to one lead."""
        for boundary in ann.findall("support/supportingROI/component/boundary"):
            code = _attr(_child(boundary, "code"), "code")
            if "_ECG_LEAD_" in code.upper():
                return code
        return ""

    def _fiducial(self, ann: ET.Element, value: ET.Element | None, sink: _AnnotationSink) -> None:
        """Record representative-beat wave boundaries (relative time, ms)."""
        wave = _attr(value, "code").upper()
        names = {
            "MDC_ECG_WAVC_PWAVE": "p",
            "MDC_ECG_WAVC_QRSWAVE": "qrs",
            "MDC_ECG_WAVC_TWAVE": "t",
        }
        if wave not in names:
            return
        boundary = _child(ann, "support/supportingROI/component/boundary")
        if boundary is None or _attr(_child(boundary, "code"), "code").upper() != "TIME_RELATIVE":
            return
        for edge, key in (("low", "onset"), ("high", "offset")):
            el = _child(boundary, f"value/{edge}")
            ms = self._convert(_attr(el, "value"), _attr(el, "unit"), "time", "boundary")
            if ms is not None:
                sink.fiducials.setdefault(f"{names[wave]}_{key}", ms)

    @staticmethod
    def _report_fiducials(fid: dict, record: ECGRecord) -> None:
        """Report representative-beat wave boundaries (ms) as stored."""
        if not fid:
            return
        record.raw_metadata["fiducials_ms"] = dict(fid)
        for name, value in fid.items():
            record.annotations[name] = f"{value:g}"

    def _convert(self, value: str, unit: str, quantity: str, code: str) -> float | None:
        v = _float(value)
        if v is None:
            return None
        if quantity == "count":
            return v
        table = {"time": _TIME_TO_MS, "rate": _RATE_TO_BPM, "angle": _ANGLE_TO_DEG}[quantity]
        if not unit:
            return v
        if unit not in table:
            self._warnings.append(f"Unit {unit!r} of {code} not recognised; value ignored")
            return None
        return v * table[unit]

    @staticmethod
    def _severity(value: ET.Element | None) -> str:
        """Severity as stored: the CE code, else the value text (upper-cased)."""
        return (_attr(value, "code") or _code_text(value)).upper()


class _AnnotationSink:
    """Values collected from the annotation sets of one author type."""

    def __init__(self) -> None:
        self.statements: list[tuple[str, str]] = []
        self.comments: list[str] = []
        self.severity = ""
        self.author = ""
        self.date: datetime | None = None
        self.confirmed = False
        self.measurements: dict[str, int] = {}
        self.beat_values: dict[str, list[float]] = {}
        self.fiducials: dict[str, float] = {}
        self.extra: dict[str, str] = {}
        self.beat_extra: dict[str, list[tuple[float, str]]] = {}
        self.lead_measurements: dict[str, dict[str, str]] = {}

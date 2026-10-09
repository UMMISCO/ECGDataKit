"""Data models for ECG records."""

from __future__ import annotations

import copy
import dataclasses
import json
import warnings
from dataclasses import dataclass, field, fields
from datetime import date, datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.exceptions import RawSamplesError


# ---------------------------------------------------------------------------
# Repr helpers
# ---------------------------------------------------------------------------

def _is_empty(value: object) -> bool:
    """Return True if value should be hidden in repr (empty or null)."""
    if value is None:
        return True
    if isinstance(value, str) and value == "":
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def _format_value(value: object) -> str:
    """Format a single value for YAML-style display."""
    if isinstance(value, datetime):
        # Keeps sub-second precision and the timezone when stored
        return value.isoformat(sep=" ")
    if isinstance(value, timedelta):
        total = value.total_seconds()
        if total < 59.95:
            return f"{total:.1f}s"
        # Round once so 3599.6 s reads "1h", not "59m 60s"
        h, rem = divmod(round(total), 3600)
        m, s = divmod(rem, 60)
        parts = [f"{n}{unit}" for n, unit in ((h, "h"), (m, "m"), (s, "s")) if n]
        return " ".join(parts)
    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            return f"{len(value)} samples ({value.dtype})"
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, list):
        return "[" + ", ".join(str(v) for v in value) + "]"
    if isinstance(value, bool):
        return str(value)
    return str(value)


def _yaml_repr(obj: object) -> str:
    """Build YAML-style repr for a dataclass, skipping empty/null fields."""
    cls_name = type(obj).__name__
    field_lines: list[str] = []
    for f in fields(obj):  # type: ignore[arg-type]
        value = getattr(obj, f.name)
        if _is_empty(value):
            continue
        field_lines.append(f"  {f.name}: {_format_value(value)}")
    if not field_lines:
        return f"{cls_name}: (empty)"
    return "\n".join([f"{cls_name}:"] + field_lines)


def _section_lines(obj: object) -> list[str]:
    """Return indented field lines for a nested dataclass section."""
    result: list[str] = []
    for f in fields(obj):  # type: ignore[arg-type]
        value = getattr(obj, f.name)
        if _is_empty(value):
            continue
        result.append(f"    {f.name}: {_format_value(value)}")
    return result


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

_UNIT_ALIASES: dict[str, str] = {
    "uV": "uV", "uv": "uV", "\u00b5V": "uV", "\u00b5v": "uV",
    "\u03bcV": "uV", "\u03bcv": "uV",
    "microvolt": "uV", "microvolts": "uV",
    "mV": "mV", "mv": "mV", "millivolt": "mV", "millivolts": "mV",
    "V": "V", "v": "V", "volt": "V", "volts": "V",
    "nV": "nV", "nv": "nV", "nanovolt": "nV", "nanovolts": "nV",
}
"""Map of recognized voltage unit strings to their canonical form."""

_TO_UV: dict[str, float] = {
    "nV": 0.001,
    "uV": 1.0,
    "mV": 1_000.0,
    "V": 1_000_000.0,
}
"""Conversion factors: multiply a value in the given unit to get microvolts."""


def _normalize_unit(unit: str) -> str | None:
    """Return the canonical voltage unit for *unit*, or ``None`` if unknown."""
    return _UNIT_ALIASES.get(unit) or _UNIT_ALIASES.get(unit.strip().lower())


def _json_default(value: object) -> object:
    """Convert numpy scalars and dates for :func:`json.dumps`."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_safe(value: object) -> object:
    """Return *value* with non-finite floats as ``None`` and numpy scalars as
    Python values. Sample lists (key ``"samples"``) are already safe and are
    not walked, which keeps this cheap on long recordings."""
    if isinstance(value, dict):
        return {k: v if k == "samples" else _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def _write_samples(fp, samples: np.ndarray, chunk: int = 65536) -> None:
    """Write *samples* to *fp* as a JSON array, non-finite values as null."""
    fp.write("[")
    for start in range(0, len(samples), chunk):
        if start:
            fp.write(",")
        part = samples[start:start + chunk]
        finite = np.isfinite(part)
        if finite.all():
            fp.write(",".join(map(repr, part.tolist())))
        else:
            fp.write(",".join(
                repr(v) if ok else "null" for v, ok in zip(part.tolist(), finite.tolist())
            ))
    fp.write("]")


def _short(text: object, width: int = 80) -> str:
    """One-line preview of a possibly long or multi-line annotation value."""
    s = str(text)
    lines = s.splitlines()
    first = lines[0] if lines else ""
    if len(first) > width:
        first = first[:width] + "..."
    if len(lines) > 1:
        first += f" ({len(lines)} lines)"
    return first


def derive_is_raw(resolution: float, offset: float, resolution_unit: str) -> bool:
    """Return ``True`` when a lead's samples are raw ADC counts.

    Samples count as already-physical (``is_raw=False``) only when the
    physical unit is known (*resolution_unit* is non-empty) **and**
    applying the scale is a no-op (``resolution == 1.0`` and
    ``offset == 0.0``).  A lead with no scaling metadata keeps the default
    ``resolution == 1.0`` but has an empty unit, so its samples are treated
    as raw ADC counts rather than mistaken for physical values.
    """
    return not (bool(resolution_unit) and resolution == 1.0 and offset == 0.0)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class PatientInfo:
    """Patient demographic information."""

    patient_id: str = ""
    """Patient identifier."""
    first_name: str = ""
    """First name."""
    last_name: str = ""
    """Last name."""
    birth_date: datetime | None = None
    """Date of birth."""
    sex: str = ""
    """Sex (``"M"``, ``"F"``, or ``"U"``)."""
    race: str = ""
    """Race/ethnicity."""
    age: int | None = None
    """Age in years."""
    weight: float | None = None
    """Weight in kg."""
    height: float | None = None
    """Height in cm."""
    medications: list[str] = field(default_factory=list)
    """Current medications."""
    clinical_history: str = ""
    """Clinical history notes."""
    has_pacemaker: bool | None = None
    """Whether the patient has a pacemaker (``None`` when not recorded)."""
    pacemaker_type: str = ""
    """Pacemaker type (e.g. ``"dual chamber bipolar"``)."""

    def __repr__(self) -> str:
        return _yaml_repr(self)

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "patient_id": self.patient_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "birth_date": self.birth_date.isoformat() if self.birth_date else None,
            "sex": self.sex,
            "race": self.race,
            "age": self.age,
            "weight": self.weight,
            "height": self.height,
            "medications": list(self.medications),
            "clinical_history": self.clinical_history,
            "has_pacemaker": self.has_pacemaker,
            "pacemaker_type": self.pacemaker_type,
        }


@dataclass
class DeviceInfo:
    """Acquisition device metadata."""

    manufacturer: str = ""
    """Device manufacturer."""
    model: str = ""
    """Device model name."""
    name: str = ""
    """Device name (distinct from model, when available)."""
    serial_number: str = ""
    """Device serial number."""
    software_version: str = ""
    """Software version."""
    institution: str = ""
    """Institution name."""
    department: str = ""
    """Department name."""
    acquisition_type: str = ""
    """Acquisition type."""

    def __repr__(self) -> str:
        return _yaml_repr(self)

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "manufacturer": self.manufacturer,
            "model": self.model,
            "name": self.name,
            "serial_number": self.serial_number,
            "software_version": self.software_version,
            "institution": self.institution,
            "department": self.department,
            "acquisition_type": self.acquisition_type,
        }


@dataclass
class FilterSettings:
    """Signal filtering applied during acquisition or processing."""

    highpass: float | None = None
    """Highpass cutoff frequency (Hz)."""
    lowpass: float | None = None
    """Lowpass cutoff frequency (Hz)."""
    notch: float | None = None
    """Notch filter frequency (Hz)."""
    notch_active: bool | None = None
    """Whether notch filter is active."""
    artifact_filter: bool | None = None
    """Whether artifact filter is active."""

    def __repr__(self) -> str:
        return _yaml_repr(self)

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "highpass": self.highpass,
            "lowpass": self.lowpass,
            "notch": self.notch,
            "notch_active": self.notch_active,
            "artifact_filter": self.artifact_filter,
        }


@dataclass
class Interpretation:
    """Machine or physician ECG interpretation."""

    statements: list[tuple[str, str]] = field(default_factory=list)
    """Interpretation text statements as ``(left, right)`` tuples.

    Each tuple contains a primary statement and an optional qualifier.
    For formats without a left/right distinction the qualifier is ``""``."""
    severity: str = ""
    """Severity (``"NORMAL"``, ``"ABNORMAL"``, ``"BORDERLINE"``)."""
    source: str = ""
    """Who produced the statements: ``"machine"`` for the device's automatic
    analysis (not a diagnosis), ``"overread"`` or ``"confirmed"`` for a
    physician. When a file holds both, the physician statements are kept here
    and the device statements go to
    ``ECGRecord.annotations["machine_interpretation"]``."""
    interpreter: str = ""
    """Physician name (if overread or confirmed)."""
    interpretation_date: datetime | None = None
    """When interpretation was made."""

    def __repr__(self) -> str:
        return _yaml_repr(self)

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "statements": [list(s) for s in self.statements],
            "severity": self.severity,
            "source": self.source,
            "interpreter": self.interpreter,
            "interpretation_date": (
                self.interpretation_date.isoformat()
                if self.interpretation_date else None
            ),
        }


@dataclass
class GlobalMeasurements:
    """Global ECG interval and axis measurements."""

    heart_rate: int | None = None
    """Heart rate (bpm)."""
    rr_interval: int | None = None
    """RR interval (ms)."""
    pr_interval: int | None = None
    """PR interval (ms)."""
    qrs_duration: int | None = None
    """QRS duration (ms)."""
    qt_interval: int | None = None
    """QT interval (ms)."""
    qtc_bazett: int | None = None
    """QTc Bazett (ms)."""
    qtc_fridericia: int | None = None
    """QTc Fridericia (ms)."""
    p_axis: int | None = None
    """P-wave axis (degrees)."""
    qrs_axis: int | None = None
    """QRS axis (degrees)."""
    t_axis: int | None = None
    """T-wave axis (degrees)."""
    qrs_count: int | None = None
    """Total QRS count."""

    def __repr__(self) -> str:
        return _yaml_repr(self)

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "heart_rate": self.heart_rate,
            "rr_interval": self.rr_interval,
            "pr_interval": self.pr_interval,
            "qrs_duration": self.qrs_duration,
            "qt_interval": self.qt_interval,
            "qtc_bazett": self.qtc_bazett,
            "qtc_fridericia": self.qtc_fridericia,
            "p_axis": self.p_axis,
            "qrs_axis": self.qrs_axis,
            "t_axis": self.t_axis,
            "qrs_count": self.qrs_count,
        }


@dataclass
class SignalCharacteristics:
    """Technical signal encoding and acquisition metadata."""

    sampling_rate: int = 0
    """Samples per second (Hz)."""
    resolution: float = 0.0
    """ADC resolution factor (e.g. µV per count)."""
    bits_per_sample: int | None = None
    """Bits per sample (e.g. 16, 12, 32)."""
    signal_offset: int | None = None
    """ADC zero/offset value."""
    signal_signed: bool | None = None
    """Whether samples are signed."""
    number_channels_allocated: int | None = None
    """Total channels in the file."""
    number_channels_valid: int | None = None
    """Channels successfully parsed."""
    electrode_placement: str = ""
    """Electrode placement code."""
    compression: str = ""
    """Compression method (e.g. ``"none"``, ``"huffman"``)."""
    data_encoding: str = ""
    """Data encoding (e.g. ``"base64_int16le"``, ``"int16"``, ``"format_212"``)."""
    acsetting: int | None = None
    """AC setting code."""
    filtered: bool | None = None
    """Whether data was pre-filtered."""
    downsampled: bool | None = None
    """Whether data was downsampled."""
    upsampled: bool | None = None
    """Whether data was upsampled."""
    waveform_modified: bool | None = None
    """Whether waveform was modified."""
    downsampling_method: str = ""
    """Downsampling method description."""
    upsampling_method: str = ""
    """Upsampling method description."""

    def __repr__(self) -> str:
        return _yaml_repr(self)

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "sampling_rate": self.sampling_rate,
            "resolution": self.resolution,
            "bits_per_sample": self.bits_per_sample,
            "signal_offset": self.signal_offset,
            "signal_signed": self.signal_signed,
            "number_channels_allocated": self.number_channels_allocated,
            "number_channels_valid": self.number_channels_valid,
            "electrode_placement": self.electrode_placement,
            "compression": self.compression,
            "data_encoding": self.data_encoding,
            "acsetting": self.acsetting,
            "filtered": self.filtered,
            "downsampled": self.downsampled,
            "upsampled": self.upsampled,
            "waveform_modified": self.waveform_modified,
            "downsampling_method": self.downsampling_method,
            "upsampling_method": self.upsampling_method,
        }


@dataclass
class AcquisitionSetup:
    """Signal acquisition configuration: characteristics and filter settings."""

    signal: SignalCharacteristics = field(default_factory=SignalCharacteristics)
    """Technical signal encoding and acquisition metadata."""
    filters: FilterSettings = field(default_factory=FilterSettings)
    """Filter settings applied during acquisition."""

    def __repr__(self) -> str:
        return _yaml_repr(self)

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "signal": self.signal.to_dict(),
            "filters": self.filters.to_dict(),
        }


@dataclass
class RecordingInfo:
    """Recording session metadata."""

    date: datetime | None = None
    """Recording start time."""
    end_date: datetime | None = None
    """Recording end time."""
    duration: timedelta | None = None
    """Recording duration."""
    technician: str = ""
    """Technician name."""
    referring_physician: str = ""
    """Referring physician name."""
    room: str = ""
    """Room identifier."""
    location: str = ""
    """Facility/location."""
    device: DeviceInfo = field(default_factory=DeviceInfo)
    """Acquisition device info."""
    acquisition: AcquisitionSetup = field(default_factory=AcquisitionSetup)
    """Signal acquisition setup (signal characteristics + filters)."""

    def __repr__(self) -> str:
        lines: list[str] = []
        cls_name = "RecordingInfo"
        # Scalar fields
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in ("device", "acquisition"):
                continue  # handled below
            if _is_empty(value):
                continue
            lines.append(f"  {f.name}: {_format_value(value)}")
        # Device sub-section
        dev_lines = _section_lines(self.device)
        if dev_lines:
            lines.append("  device:")
            for dl in dev_lines:
                lines.append(f"  {dl}")
        # Acquisition sub-section
        sig_lines = _section_lines(self.acquisition.signal)
        fil_lines = _section_lines(self.acquisition.filters)
        if sig_lines or fil_lines:
            lines.append("  acquisition:")
            if sig_lines:
                lines.append("      signal:")
                for sl in sig_lines:
                    lines.append(f"    {sl}")
            if fil_lines:
                lines.append("      filters:")
                for fl in fil_lines:
                    lines.append(f"    {fl}")
        if not lines:
            return f"{cls_name}: (empty)"
        return "\n".join([f"{cls_name}:"] + lines)

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "date": self.date.isoformat() if self.date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "duration_seconds": self.duration.total_seconds() if self.duration else None,
            "technician": self.technician,
            "referring_physician": self.referring_physician,
            "room": self.room,
            "location": self.location,
            "device": self.device.to_dict(),
            "acquisition": self.acquisition.to_dict(),
        }


@dataclass
class Lead:
    """Single ECG lead with signal data.

    **Resolution and scaling**

    ECG file formats store a raw ADC resolution value in format-specific
    units (e.g. nV/count for ISHNE and SCP-ECG, µV/count for Sierra XML).
    The parser converts this to a normalised scale factor stored in
    ``resolution``, expressed in the unit given by ``resolution_unit``:

    .. code-block:: text

        physical_value = samples * resolution + offset   (in ``resolution_unit``)

    ``units`` is the unit of ``samples`` as they are now: empty while the
    samples are raw counts, ``resolution_unit`` after :meth:`to_physical`.

    The original, unconverted value from the file is preserved in
    ``adc_resolution`` for reference.

    **Example** — ISHNE file with ``ampl_res = 153`` (nV/count):

    * ``adc_resolution = 153.0`` — raw file value (nV/count)
    * ``resolution = 0.153`` — converted: 153 / 1000 (µV/count)
    * ``units = "uV"``

    **Auto-detection of** ``is_raw``

    Parsers set ``is_raw`` via :func:`derive_is_raw`.  Samples are already
    in physical units (``is_raw=False``) only when the unit is known
    (``resolution_unit`` is set) **and** scaling is a no-op
    (``resolution == 1.0`` and ``offset == 0.0``).  Leads with a real scale
    factor, or with no scaling metadata at all, are raw ADC counts
    (``is_raw=True``) that need scaling via :meth:`to_physical`.
    """

    label: str
    """Lead name (e.g. ``"I"``, ``"V1"``)."""
    samples: NDArray[np.float64]
    """Signal sample values (raw ADC or physical, depending on ``is_raw``)."""
    sampling_rate: int
    """Samples per second (Hz)."""
    resolution: float = 1.0
    """Normalised scale factor for ADC-to-physical conversion, in the unit
    given by ``resolution_unit``.  Computed from ``adc_resolution`` by the
    parser (e.g. ``adc_resolution / 1000`` for nV → µV).  Used by
    :meth:`to_physical`: ``physical = samples * resolution + offset``."""
    resolution_unit: str = ""
    """Unit of the ``resolution`` scale factor (e.g. ``"uV"``, ``"mV"``).
    After :meth:`to_physical`, the resulting samples are in this unit.
    Set by the parser based on the format specification."""
    offset: float = 0.0
    """Additive offset for ADC-to-physical conversion (default ``0.0``).
    Used by :meth:`to_physical`: ``physical = samples * resolution + offset``."""
    units: str = ""
    """Current unit of ``samples``.  Empty when ``is_raw=True`` (samples
    are dimensionless ADC counts).  Set to the physical unit after
    :meth:`to_physical` or :meth:`convert_units` is called (e.g. ``"uV"``,
    ``"mV"``)."""
    is_raw: bool = True
    """``True`` if samples are raw ADC counts needing scaling, ``False``
    if samples are already in physical ``units``.  Parsers set this via
    :func:`derive_is_raw`: physical only when the unit is known
    (``resolution_unit`` set) and scaling is a no-op; leads without scaling
    metadata are raw even though their default ``resolution`` is ``1.0``."""
    adc_resolution: float = 0.0
    """Original ADC resolution exactly as stored in the source file,
    before any unit conversion.  For example, ISHNE stores nV/count and
    SCP-ECG stores nV/unit — this field preserves that raw value
    (e.g. ``153.0`` for 153 nV/count).  The converted value used for
    scaling is in ``resolution``."""
    adc_resolution_unit: str = ""
    """Unit of ``adc_resolution`` as defined by the source format
    (e.g. ``"nV"`` for ISHNE and SCP-ECG)."""
    quality: int | None = None
    """Signal quality indicator (format-specific)."""
    transducer: str = ""
    """Transducer type."""
    prefiltering: str = ""
    """Pre-filtering description."""
    annotations: dict[str, str] = field(default_factory=dict)
    """Per-lead measurements/annotations (format-specific key-value pairs)."""

    def __repr__(self) -> str:
        lines = ["Lead:"]
        lines.append(f"  label: {self.label}")
        n = len(self.samples)
        sr = self.sampling_rate
        dur = f" ({n / sr:.1f}s)" if sr else ""
        lines.append(f"  samples: {n} samples{dur}")
        lines.append(f"  sampling_rate: {sr}")
        lines.append(f"  is_raw: {self.is_raw}")
        lines.append(f"  resolution: {self.resolution}")
        if self.resolution_unit:
            lines.append(f"  resolution_unit: {self.resolution_unit}")
        if self.offset != 0.0:
            lines.append(f"  offset: {self.offset}")
        if self.units:
            lines.append(f"  units: {self.units}")
        if self.adc_resolution != 0.0:
            lines.append(f"  adc_resolution: {self.adc_resolution}")
        if self.adc_resolution_unit:
            lines.append(f"  adc_resolution_unit: {self.adc_resolution_unit}")
        if self.quality is not None:
            lines.append(f"  quality: {self.quality}")
        if self.transducer:
            lines.append(f"  transducer: {self.transducer}")
        if self.prefiltering:
            lines.append(f"  prefiltering: {self.prefiltering}")
        if self.annotations:
            lines.append(f"  annotations: {len(self.annotations)} entries")
        return "\n".join(lines)

    def to_physical(self) -> Lead:
        """Convert raw ADC samples to physical voltage units.

        Applies ``physical = samples * resolution + offset`` and returns
        a **new** :class:`Lead` with ``is_raw=False`` and a new sample
        array.  If this lead is already in physical units, returns ``self``
        (same object, same array).

        Raises
        ------
        ValueError
            If ``resolution`` is zero (conversion undefined).
        RawSamplesError
            If the file gives no physical unit (``resolution_unit`` is
            empty): the samples are uncalibrated counts and cannot be
            converted.
        """
        if not self.is_raw:
            return self
        if self.resolution == 0.0:
            raise ValueError(
                f"Lead '{self.label}': resolution is 0, "
                "cannot convert to physical units"
            )
        if not self.resolution_unit:
            raise RawSamplesError(
                f"Lead '{self.label}': no physical unit in the file, the "
                "samples are uncalibrated counts and cannot be converted"
            )
        return dataclasses.replace(
            self,
            samples=self.samples * self.resolution + self.offset,
            is_raw=False,
            units=self.resolution_unit,
        )

    def convert_units(self, target: str) -> Lead:
        """Convert between physical voltage units (uV, mV, V).

        Parameters
        ----------
        target : str
            Target unit string (``"uV"``, ``"mV"``, ``"V"`` and common
            aliases like ``"\u00b5V"``).

        Returns
        -------
        Lead
            A new :class:`Lead` with a new, scaled sample array, or ``self``
            when the lead is already in *target*.

        Raises
        ------
        RawSamplesError
            If samples are still raw ADC (``is_raw=True``).
        ValueError
            If the current or target unit is not a recognized voltage unit.
        """
        if self.is_raw:
            raise RawSamplesError(
                f"Lead '{self.label}': cannot convert units on raw ADC "
                "samples. Call to_physical() first."
            )
        target_norm = _normalize_unit(target)
        if target_norm is None:
            raise ValueError(
                f"Unknown target unit '{target}'. "
                "Accepted units: uV, mV, V (and aliases)."
            )
        current_norm = _normalize_unit(self.units)
        if current_norm is None:
            raise ValueError(
                f"Lead '{self.label}': current unit '{self.units}' is not "
                "a recognized voltage unit. Cannot convert."
            )
        if current_norm == target_norm:
            return self
        factor = _TO_UV[current_norm] / _TO_UV[target_norm]
        return dataclasses.replace(
            self,
            samples=self.samples * factor,
            units=target_norm,
        )

    def to_dict(self, include_samples: bool = True) -> dict:
        """Convert to a JSON-serialisable dictionary.

        Parameters
        ----------
        include_samples : bool
            If ``True`` (default), include the full sample array.
            Set to ``False`` for a lightweight summary.
        """
        d: dict = {
            "label": self.label,
            "sample_count": len(self.samples),
            "sampling_rate": self.sampling_rate,
            "resolution": self.resolution,
            "resolution_unit": self.resolution_unit,
            "offset": self.offset,
            "units": self.units,
            "is_raw": self.is_raw,
            "adc_resolution": self.adc_resolution,
            "adc_resolution_unit": self.adc_resolution_unit,
            "quality": self.quality,
            "transducer": self.transducer,
            "prefiltering": self.prefiltering,
            "annotations": dict(self.annotations),
        }
        if include_samples:
            samples = self.samples
            if not np.isfinite(samples).all():
                # JSON has no NaN/Inf, write them as null
                samples = np.where(np.isfinite(samples), samples, None)
            d["samples"] = samples.tolist()
        return d


LeadLike = Lead | NDArray[np.float64]
"""Type alias: accepts a :class:`Lead` or a raw numpy array of samples."""

LeadsLike = "list[Lead] | ECGRecord | NDArray[np.float64] | list[NDArray[np.float64]]"
"""Type alias: accepts a list of :class:`Lead`, an :class:`ECGRecord`,
a 2-D numpy array (n_leads × n_samples), or a list of 1-D numpy arrays."""


@dataclass
class FileFormatInfo:
    """Source file format details."""

    version: str = ""
    """Format version declared in the file."""
    creation_date: date | None = None
    """Date the file was written."""

    def __repr__(self) -> str:
        return _yaml_repr(self)

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "version": self.version,
            "creation_date": self.creation_date.isoformat() if self.creation_date else None,
        }


@dataclass
class ECGRecord:
    """Unified ECG record returned by all parsers.

    Every parser in ECGDataKit produces an ``ECGRecord``.  Use
    :meth:`to_dict` or :meth:`to_json` to obtain a format-agnostic,
    JSON-serialisable representation that is identical regardless of the
    original file format.

    Samples are stored as raw ADC values by default.  Call
    :meth:`to_physical` to convert all leads to physical voltage units,
    then :meth:`convert_units` to switch between ``uV``, ``mV``, or ``V``.
    """

    patient: PatientInfo = field(default_factory=PatientInfo)
    """Patient demographics."""
    recording: RecordingInfo = field(default_factory=RecordingInfo)
    """Recording session metadata (includes device and acquisition setup)."""
    leads: list[Lead] = field(default_factory=list)
    """ECG lead waveforms."""
    interpretation: Interpretation = field(default_factory=Interpretation)
    """Machine or physician interpretation."""
    measurements: GlobalMeasurements = field(default_factory=GlobalMeasurements)
    """Global ECG interval/axis measurements."""
    median_beats: list[Lead] = field(default_factory=list)
    """Median/template beats if available."""
    annotations: dict[str, str] = field(default_factory=dict)
    """Values the file stores that have no field in the model, one value per
    key. A value that fits ``measurements`` is stored there, not here. Wave
    boundaries of the representative beat start with ``median_`` (for example
    ``median_p_onset_ms``). Keys end with their unit when the file or its
    specification states it (``_ms``, ``_samples``, ``_s``)."""
    source_format: str = ""
    """Parser identifier (e.g. ``"hl7_aecg"``, ``"dicom"``)."""
    raw_metadata: dict = field(default_factory=dict)
    """Original format-specific metadata from the source file."""
    file_format: FileFormatInfo = field(default_factory=FileFormatInfo)
    """Source file format version and creation date."""
    leads_enhanced: list[Lead] = field(default_factory=list)
    """The same leads as filtered by the device, when the file also stores
    that version (AliveCor ``enhanced``). Empty for other formats."""

    def __repr__(self) -> str:
        lines = ["ECGRecord:"]

        if self.source_format:
            lines.append(f"  source_format: {self.source_format}")

        flines = _section_lines(self.file_format)
        if flines:
            lines.append("  file_format:")
            lines.extend(flines)

        # Patient section
        plines = _section_lines(self.patient)
        if plines:
            lines.append("  patient:")
            lines.extend(plines)

        # Recording section (includes device + acquisition sub-sections)
        rec = self.recording
        rec_scalar: list[str] = []
        for f in fields(rec):
            if f.name in ("device", "acquisition"):
                continue
            value = getattr(rec, f.name)
            if _is_empty(value):
                continue
            rec_scalar.append(f"    {f.name}: {_format_value(value)}")
        dev_lines = _section_lines(rec.device)
        sig_lines = _section_lines(rec.acquisition.signal)
        fil_lines = _section_lines(rec.acquisition.filters)
        if rec_scalar or dev_lines or sig_lines or fil_lines:
            lines.append("  recording:")
            lines.extend(rec_scalar)
            if dev_lines:
                lines.append("    device:")
                for dl in dev_lines:
                    lines.append(f"  {dl}")
            if sig_lines or fil_lines:
                lines.append("    acquisition:")
                if sig_lines:
                    lines.append("      signal:")
                    for sl in sig_lines:
                        lines.append(f"    {sl}")
                if fil_lines:
                    lines.append("      filters:")
                    for fl in fil_lines:
                        lines.append(f"    {fl}")

        # Measurements / interpretation
        for name, obj in [("measurements", self.measurements), ("interpretation", self.interpretation)]:
            slines = _section_lines(obj)
            if slines:
                lines.append(f"  {name}:")
                lines.extend(slines)

        # Leads
        if self.leads:
            lines.append("  leads:")
            for lead in self.leads:
                n = len(lead.samples)
                sr = lead.sampling_rate
                dur = f", {n / sr:.1f}s" if sr else ""
                status = "raw" if lead.is_raw else (lead.units or "physical")
                lines.append(
                    f"    - {lead.label}: {n} samples, {sr} Hz{dur}, {status}"
                )

        if self.leads_enhanced:
            lines.append("  leads_enhanced:")
            for lead in self.leads_enhanced:
                status = "raw" if lead.is_raw else (lead.units or "physical")
                lines.append(
                    f"    - {lead.label}: {len(lead.samples)} samples, "
                    f"{lead.sampling_rate} Hz, {status}"
                )

        # Median beats
        if self.median_beats:
            lines.append("  median_beats:")
            for beat in self.median_beats:
                lines.append(
                    f"    - {beat.label}: {len(beat.samples)} samples"
                )

        # Annotations
        if self.annotations:
            lines.append("  annotations:")
            for k, v in self.annotations.items():
                lines.append(f"    {k}: {_short(v)}")

        # Raw metadata indicator
        if self.raw_metadata:
            lines.append(f"  raw_metadata: {len(self.raw_metadata)} entries")

        return "\n".join(lines)

    def _with_signals(
        self, leads: list[Lead], median_beats: list[Lead], leads_enhanced: list[Lead],
    ) -> ECGRecord:
        """Return a copy with new lead lists and its own copy of the metadata."""
        meta = copy.deepcopy(dataclasses.replace(
            self, leads=[], median_beats=[], leads_enhanced=[]))
        return dataclasses.replace(
            meta, leads=leads, median_beats=median_beats, leads_enhanced=leads_enhanced)

    def to_physical(self) -> ECGRecord:
        """Convert all leads, median beats and enhanced leads from raw ADC to
        physical units.

        Returns a new :class:`ECGRecord` with its own copy of the metadata.
        Leads already in physical units keep their sample array (it is
        shared with this record). Leads whose file gives no physical unit
        (uncalibrated counts) are left raw and reported in a warning.
        """
        skipped: list[str] = []

        def convert(lead: Lead) -> Lead:
            if lead.is_raw and lead.resolution != 0.0 and not lead.resolution_unit:
                skipped.append(lead.label)
                return lead
            return lead.to_physical()

        leads = [convert(lead) for lead in self.leads]
        beats = [convert(beat) for beat in self.median_beats]
        enhanced = [convert(lead) for lead in self.leads_enhanced]
        if skipped:
            warnings.warn(
                f"Leads {skipped} have no physical unit in the file and were "
                "left as raw counts.",
                stacklevel=2,
            )
        return self._with_signals(leads, beats, enhanced)

    def convert_units(self, target: str) -> ECGRecord:
        """Convert all leads, median beats and enhanced leads to the specified
        voltage unit.

        Returns a new :class:`ECGRecord` with its own copy of the metadata.
        Leads already in *target* keep their sample array (shared with this
        record).

        Parameters
        ----------
        target : str
            Target unit (``"uV"``, ``"mV"``, ``"V"``).

        Raises
        ------
        RawSamplesError
            If any lead is still raw ADC.
        """
        return self._with_signals(
            [lead.convert_units(target) for lead in self.leads],
            [beat.convert_units(target) for beat in self.median_beats],
            [lead.convert_units(target) for lead in self.leads_enhanced],
        )

    def plot(
        self,
        show: bool = True,
        rows: int | None = None,
        cols: int | None = None,
        **kwargs,
    ):
        """Plot the ECG record with patient/device header and all leads.

        Parameters
        ----------
        show : bool
            Display the plot immediately (default ``True``).
        rows : int | None
            Number of rows in the subplot grid.
        cols : int | None
            Number of columns in the subplot grid.
        **kwargs
            Extra arguments forwarded to the underlying plot function
            (e.g. ``figsize``, ``x_axis``).
        """
        from ecgdatakit.plotting.static import plot_12lead, plot_leads

        if len(self.leads) >= 12:
            return plot_12lead(self, record=self, show=show, rows=rows, cols=cols, **kwargs)
        return plot_leads(self, show=show, rows=rows, cols=cols, **kwargs)

    def to_dict(self, include_samples: bool = True) -> dict:
        """Convert the record to the **unified JSON schema**.

        The result is JSON-safe: non-finite numbers (NaN, inf) are ``None``.
        With samples, every value becomes a Python float, which takes about
        30 bytes per sample (a 24 h 12-lead Holter at 180 Hz needs ~6 GB).

        Parameters
        ----------
        include_samples : bool
            If ``True`` (default), each lead contains its full sample
            array.  Set to ``False`` for metadata-only export.
        """
        return _json_safe({
            "source_format": self.source_format,
            "file_format": self.file_format.to_dict(),
            "patient": self.patient.to_dict(),
            "recording": self.recording.to_dict(),
            "leads": [lead.to_dict(include_samples=include_samples) for lead in self.leads],
            "interpretation": self.interpretation.to_dict(),
            "measurements": self.measurements.to_dict(),
            "median_beats": [b.to_dict(include_samples=include_samples) for b in self.median_beats],
            "leads_enhanced": [
                lead.to_dict(include_samples=include_samples) for lead in self.leads_enhanced
            ],
            "annotations": dict(self.annotations),
        })

    def to_json(
        self,
        include_samples: bool = True,
        indent: int | None = 2,
        fp=None,
    ) -> str | None:
        """Serialise the record to JSON (strict: no NaN or Infinity).

        Parameters
        ----------
        include_samples : bool
            Include full sample arrays (default ``True``).
        indent : int | None
            JSON indentation level.  ``None`` for compact output.  Ignored
            when *fp* is given.
        fp : file-like, optional
            Text stream to write to.  The JSON is then written compactly,
            lead by lead and in chunks of samples, and ``None`` is returned.
            Use it for long recordings: building the string in memory takes
            several times the size of the output (a 24 h 12-lead Holter
            gives ~2 GB of JSON and needs ~12 GB as a string).
        """
        if fp is None:
            return json.dumps(
                self.to_dict(include_samples=include_samples),
                indent=indent,
                default=_json_default,
                allow_nan=False,
            )
        meta = self.to_dict(include_samples=False)
        dump = lambda v: json.dumps(v, default=_json_default, allow_nan=False)  # noqa: E731
        fp.write("{")
        for i, (key, value) in enumerate(meta.items()):
            if i:
                fp.write(",")
            fp.write(dump(key) + ":")
            if key not in ("leads", "median_beats", "leads_enhanced") or not include_samples:
                fp.write(dump(value))
                continue
            signals = {"leads": self.leads, "median_beats": self.median_beats,
                       "leads_enhanced": self.leads_enhanced}[key]
            fp.write("[")
            for j, (lead_meta, lead) in enumerate(zip(value, signals)):
                if j:
                    fp.write(",")
                fp.write(dump(lead_meta)[:-1] + ',"samples":')
                _write_samples(fp, lead.samples)
                fp.write("}")
            fp.write("]")
        fp.write("}")
        return None

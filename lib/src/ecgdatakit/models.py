"""Data models for ECG records."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta

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
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, timedelta):
        total = value.total_seconds()
        if total >= 3600:
            h, rem = divmod(total, 3600)
            m, s = divmod(rem, 60)
            parts = [f"{int(h)}h"]
            if m:
                parts.append(f"{int(m)}m")
            if s:
                parts.append(f"{s:.0f}s")
            return " ".join(parts)
        if total >= 60:
            m, s = divmod(total, 60)
            parts = [f"{int(m)}m"]
            if s:
                parts.append(f"{s:.0f}s")
            return " ".join(parts)
        return f"{total:.1f}s"
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
    "microvolt": "uV", "microvolts": "uV",
    "mV": "mV", "mv": "mV", "millivolt": "mV", "millivolts": "mV",
    "V": "V", "v": "V", "volt": "V", "volts": "V",
}
"""Map of recognized voltage unit strings to their canonical form."""

_TO_UV: dict[str, float] = {
    "uV": 1.0,
    "mV": 1_000.0,
    "V": 1_000_000.0,
}
"""Conversion factors: multiply a value in the given unit to get microvolts."""


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
    """Source (``"machine"``, ``"overread"``, ``"confirmed"``)."""
    interpreter: str = ""
    """Physician name (if overread)."""
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

    sample_rate: int = 0
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
            "sample_rate": self.sample_rate,
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

    Parsers auto-detect whether samples are raw ADC or already in
    physical units based on the scaling metadata (``resolution`` and
    ``offset``).  When ``is_raw=True``, call :meth:`to_physical` to
    convert to physical voltage units.
    """

    label: str
    """Lead name (e.g. ``"I"``, ``"V1"``)."""
    samples: NDArray[np.float64]
    """Signal sample values (raw ADC or physical, depending on ``is_raw``)."""
    sample_rate: int
    """Samples per second (Hz)."""
    resolution: float = 1.0
    """Scale factor for ADC-to-physical conversion (default ``1.0``)."""
    offset: float = 0.0
    """Additive offset for ADC-to-physical conversion (default ``0.0``)."""
    units: str = ""
    """Physical voltage unit (e.g. ``"mV"``, ``"uV"``).  When ``is_raw=True``
    this is the target unit that ``resolution`` converts to; when
    ``is_raw=False`` it is the actual unit of ``samples``."""
    is_raw: bool = True
    """``True`` if samples are raw ADC counts needing scaling, ``False``
    if samples are already in physical ``units``.  Parsers set this
    automatically based on ``resolution`` and ``offset``."""
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
        sr = self.sample_rate
        dur = f" ({n / sr:.1f}s)" if sr else ""
        lines.append(f"  samples: {n} samples{dur}")
        lines.append(f"  sample_rate: {sr}")
        lines.append(f"  is_raw: {self.is_raw}")
        lines.append(f"  resolution: {self.resolution}")
        if self.offset != 0.0:
            lines.append(f"  offset: {self.offset}")
        if self.units:
            lines.append(f"  units: {self.units}")
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
        a **new** :class:`Lead` with ``is_raw=False``.  If this lead is
        already in physical units, returns ``self`` unchanged.

        Raises
        ------
        ValueError
            If ``resolution`` is zero (conversion undefined).
        """
        if not self.is_raw:
            return self
        if self.resolution == 0.0:
            raise ValueError(
                f"Lead '{self.label}': resolution is 0, "
                "cannot convert to physical units"
            )
        return dataclasses.replace(
            self,
            samples=self.samples * self.resolution + self.offset,
            is_raw=False,
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
            A new :class:`Lead` with samples scaled to *target*.

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
        target_norm = _UNIT_ALIASES.get(target)
        if target_norm is None:
            raise ValueError(
                f"Unknown target unit '{target}'. "
                "Accepted units: uV, mV, V (and aliases)."
            )
        current_norm = _UNIT_ALIASES.get(self.units)
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
            "sample_rate": self.sample_rate,
            "resolution": self.resolution,
            "offset": self.offset,
            "units": self.units,
            "is_raw": self.is_raw,
            "quality": self.quality,
            "transducer": self.transducer,
            "prefiltering": self.prefiltering,
            "annotations": dict(self.annotations),
        }
        if include_samples:
            d["samples"] = self.samples.tolist()
        return d


LeadLike = Lead | NDArray[np.float64]
"""Type alias: accepts a :class:`Lead` or a raw numpy array of samples."""

LeadsLike = "list[Lead] | ECGRecord | NDArray[np.float64] | list[NDArray[np.float64]]"
"""Type alias: accepts a list of :class:`Lead`, an :class:`ECGRecord`,
a 2-D numpy array (n_leads × n_samples), or a list of 1-D numpy arrays."""


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
    """Additional key-value annotations."""
    source_format: str = ""
    """Parser identifier (e.g. ``"hl7_aecg"``, ``"dicom"``)."""
    raw_metadata: dict = field(default_factory=dict)
    """Original format-specific metadata from the source file."""

    def __repr__(self) -> str:
        lines = ["ECGRecord:"]

        if self.source_format:
            lines.append(f"  source_format: {self.source_format}")

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
            lines.append(f"  leads:")
            for lead in self.leads:
                n = len(lead.samples)
                sr = lead.sample_rate
                dur = f", {n / sr:.1f}s" if sr else ""
                status = "raw" if lead.is_raw else (lead.units or "physical")
                lines.append(
                    f"    - {lead.label}: {n} samples, {sr} Hz{dur}, {status}"
                )

        # Median beats
        if self.median_beats:
            lines.append(f"  median_beats:")
            for beat in self.median_beats:
                lines.append(
                    f"    - {beat.label}: {len(beat.samples)} samples"
                )

        # Annotations
        if self.annotations:
            lines.append(f"  annotations:")
            for k, v in self.annotations.items():
                lines.append(f"    {k}: {v}")

        # Raw metadata indicator
        if self.raw_metadata:
            lines.append(f"  raw_metadata: {len(self.raw_metadata)} entries")

        return "\n".join(lines)

    def to_physical(self) -> ECGRecord:
        """Convert all leads and median beats from raw ADC to physical units.

        Returns a new :class:`ECGRecord` where every :class:`Lead` has
        ``is_raw=False``.  Leads already in physical units are unchanged.
        """
        return dataclasses.replace(
            self,
            leads=[lead.to_physical() for lead in self.leads],
            median_beats=[beat.to_physical() for beat in self.median_beats],
        )

    def convert_units(self, target: str) -> ECGRecord:
        """Convert all leads and median beats to the specified voltage unit.

        Parameters
        ----------
        target : str
            Target unit (``"uV"``, ``"mV"``, ``"V"``).

        Raises
        ------
        RawSamplesError
            If any lead is still raw ADC.
        """
        return dataclasses.replace(
            self,
            leads=[lead.convert_units(target) for lead in self.leads],
            median_beats=[beat.convert_units(target) for beat in self.median_beats],
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

        Parameters
        ----------
        include_samples : bool
            If ``True`` (default), each lead contains its full sample
            array.  Set to ``False`` for metadata-only export.
        """
        return {
            "source_format": self.source_format,
            "patient": self.patient.to_dict(),
            "recording": self.recording.to_dict(),
            "leads": [lead.to_dict(include_samples=include_samples) for lead in self.leads],
            "interpretation": self.interpretation.to_dict(),
            "measurements": self.measurements.to_dict(),
            "median_beats": [b.to_dict(include_samples=include_samples) for b in self.median_beats],
            "annotations": dict(self.annotations),
        }

    def to_json(self, include_samples: bool = True, indent: int | None = 2) -> str:
        """Serialise the record to a JSON string.

        Parameters
        ----------
        include_samples : bool
            Include full sample arrays (default ``True``).
        indent : int | None
            JSON indentation level.  ``None`` for compact output.
        """
        return json.dumps(self.to_dict(include_samples=include_samples), indent=indent)

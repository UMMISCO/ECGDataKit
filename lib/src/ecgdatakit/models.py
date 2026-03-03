"""Data models for ECG records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray


@dataclass
class PatientInfo:
    """Patient demographic information."""

    patient_id: str = ""
    first_name: str = ""
    last_name: str = ""
    birth_date: datetime | None = None
    sex: str = ""
    race: str = ""
    age: int | None = None
    weight: float | None = None
    height: float | None = None
    medications: list[str] = field(default_factory=list)
    clinical_history: str = ""

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
    model: str = ""
    name: str = ""
    serial_number: str = ""
    software_version: str = ""
    institution: str = ""
    department: str = ""
    acquisition_type: str = ""

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
    lowpass: float | None = None
    notch: float | None = None
    notch_active: bool | None = None
    artifact_filter: bool | None = None

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

    statements: list[str] = field(default_factory=list)
    severity: str = ""
    source: str = ""
    interpreter: str = ""
    interpretation_date: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "statements": list(self.statements),
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
    rr_interval: int | None = None
    pr_interval: int | None = None
    qrs_duration: int | None = None
    qt_interval: int | None = None
    qtc_bazett: int | None = None
    qtc_fridericia: int | None = None
    p_axis: int | None = None
    qrs_axis: int | None = None
    t_axis: int | None = None
    qrs_count: int | None = None

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

    bits_per_sample: int | None = None
    signal_offset: int | None = None
    signal_signed: bool | None = None
    number_channels_allocated: int | None = None
    number_channels_valid: int | None = None
    electrode_placement: str = ""
    compression: str = ""
    data_encoding: str = ""
    acsetting: int | None = None
    filtered: bool | None = None
    downsampled: bool | None = None
    upsampled: bool | None = None
    waveform_modified: bool | None = None
    downsampling_method: str = ""
    upsampling_method: str = ""

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dictionary."""
        return {
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
class RecordingInfo:
    """Recording session metadata."""

    date: datetime | None = None
    end_date: datetime | None = None
    duration: timedelta | None = None
    sample_rate: int = 0
    adc_gain: float = 1.0
    technician: str = ""
    referring_physician: str = ""
    room: str = ""
    location: str = ""

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "date": self.date.isoformat() if self.date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "duration_seconds": self.duration.total_seconds() if self.duration else None,
            "sample_rate": self.sample_rate,
            "adc_gain": self.adc_gain,
            "technician": self.technician,
            "referring_physician": self.referring_physician,
            "room": self.room,
            "location": self.location,
        }


@dataclass
class Lead:
    """Single ECG lead with signal data."""

    label: str
    samples: NDArray[np.float64]
    sample_rate: int
    resolution: float = 1.0
    units: str = ""
    quality: int | None = None
    transducer: str = ""
    prefiltering: str = ""

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
            "units": self.units,
            "quality": self.quality,
            "transducer": self.transducer,
            "prefiltering": self.prefiltering,
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
    """

    patient: PatientInfo = field(default_factory=PatientInfo)
    recording: RecordingInfo = field(default_factory=RecordingInfo)
    device: DeviceInfo = field(default_factory=DeviceInfo)
    filters: FilterSettings = field(default_factory=FilterSettings)
    signal: SignalCharacteristics = field(default_factory=SignalCharacteristics)
    leads: list[Lead] = field(default_factory=list)
    interpretation: Interpretation = field(default_factory=Interpretation)
    measurements: GlobalMeasurements = field(default_factory=GlobalMeasurements)
    median_beats: list[Lead] = field(default_factory=list)
    annotations: dict[str, str] = field(default_factory=dict)
    source_format: str = ""
    raw_metadata: dict = field(default_factory=dict)

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
            "device": self.device.to_dict(),
            "filters": self.filters.to_dict(),
            "signal": self.signal.to_dict(),
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

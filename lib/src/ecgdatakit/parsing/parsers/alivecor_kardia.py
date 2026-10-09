"""AliveCor Kardia JSON parser.

Reads the ECG recording object returned by the Kardia API
(``GET /v1/recordings/:id``), for KardiaMobile single-lead and 6L devices.
Field meanings follow the Kardia API documentation
(https://developers.kardia.com):

- ``data.raw`` and ``data.enhanced``: the unfiltered signal and the same
  signal filtered by AliveCor. Each holds ``frequency`` (Hz),
  ``mainsFrequency`` (50 or 60 Hz), ``amplitudeResolution`` (nanovolts per
  sample unit) and ``samples`` (``leadI``, ``leadII``, ``leadIII``, ``AVR``,
  ``AVL``, ``AVF``) in ADC units.
- ``duration`` in milliseconds, ``heartRate`` average heart rate,
  ``recordedAt`` ISO 8601 date and time.
- ``algorithmDetermination``: output of the AliveCor algorithms (normal,
  afib, unclassified, bradycardia, tachycardia, too_short, too_long,
  unreadable, no_analysis).
- ``deviceInfo``: hardwareType, hardwareRevision, firmwareRevision,
  serialNumber.
- ``qtMeasurement``: qt, rr, qtcb, qtcf, each as ``{"Value": ...}``.

Kardia 12L recordings (``data12L``) use another layout with no stated
amplitude resolution and are not supported.
"""

from __future__ import annotations

import json
import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ecgdatakit.exceptions import CorruptedFileError, UnsupportedFormatError
from ecgdatakit.models import (
    DeviceInfo,
    ECGRecord,
    GlobalMeasurements,
    Interpretation,
    Lead,
    SignalCharacteristics,
    derive_is_raw,
)
from ecgdatakit.parsing.helpers import fill_signal_summary, normalize_lead_label, unique_labels
from ecgdatakit.parsing.parser import Parser

_KEY_RE = re.compile(r'"(recordedAt|algorithmDetermination|patientID|amplitudeResolution|data12L)"')

_LEAD_NAMES = {"leadI": "I", "leadII": "II", "leadIII": "III",
               "AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}

# qtMeasurement keys and the matching model fields
_QT_FIELDS = {"qt": "qt_interval", "rr": "rr_interval",
              "qtcb": "qtc_bazett", "qtcf": "qtc_fridericia"}


def _text(value) -> str:
    return "" if value is None else str(value).strip()


def _number(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if np.isfinite(value) else None
    try:
        number = float(_text(value))
    except ValueError:
        return None
    return number if np.isfinite(number) else None


class AliveCorKardiaParser(Parser):
    """Parser for AliveCor Kardia API ECG recordings (JSON)."""

    FORMAT_NAME = "AliveCor Kardia JSON"
    FORMAT_DESCRIPTION = "AliveCor Kardia API ECG recording (KardiaMobile, KardiaMobile 6L)"
    FILE_EXTENSIONS = [".json"]
    PRIORITY = 60

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        text = header.decode("utf-8", errors="ignore").lstrip("﻿ \t\r\n")
        if not text.startswith("{"):
            return False
        keys = set(_KEY_RE.findall(text))
        return "recordedAt" in keys and len(keys) >= 2

    def parse(self, file_path: Path) -> ECGRecord:
        try:
            doc = json.loads(Path(file_path).read_bytes())
        except (ValueError, UnicodeDecodeError) as e:
            raise CorruptedFileError(f"Malformed Kardia JSON in {file_path}: {e}") from e
        if not isinstance(doc, dict):
            raise CorruptedFileError(f"{file_path}: expected a Kardia recording object")
        if "data12L" in doc:
            raise UnsupportedFormatError(
                f"{file_path}: Kardia 12L recordings are not supported"
            )
        data = doc.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("raw"), dict):
            raise CorruptedFileError(f"{file_path}: no ECG data (data.raw) in the recording")

        record = ECGRecord(source_format="alivecor_kardia")
        raw_meta = record.raw_metadata

        record.leads, raw_info = self._read_block(data["raw"], "raw", file_path)
        if isinstance(data.get("enhanced"), dict):
            record.leads_enhanced, enhanced_info = self._read_block(
                data["enhanced"], "enhanced", file_path)
            raw_meta["enhanced"] = enhanced_info
        raw_meta["raw"] = raw_info

        record.patient.patient_id = _text(doc.get("patientID"))

        recording = record.recording
        recorded_at = _text(doc.get("recordedAt"))
        if recorded_at:
            try:
                recording.date = datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
            except ValueError:
                warnings.warn(f"Kardia recordedAt {recorded_at!r} is not a valid date",
                              stacklevel=2)
                raw_meta["recorded_at"] = recorded_at
        duration = _number(doc.get("duration"))
        if duration is not None and duration > 0:
            recording.duration = timedelta(milliseconds=duration)

        device = doc.get("deviceInfo") if isinstance(doc.get("deviceInfo"), dict) else {}
        info = {k.lower(): _text(v) for k, v in device.items()}
        recording.device = DeviceInfo(
            manufacturer="AliveCor",
            model=info.get("hardwaretype", ""),
            serial_number=info.get("serialnumber", ""),
            software_version=info.get("firmwarerevision", ""),
        )
        if info.get("hardwarerevision"):
            record.annotations["hardware_revision"] = info["hardwarerevision"]

        first = record.leads[0]
        recording.acquisition.signal = SignalCharacteristics(
            sampling_rate=first.sampling_rate,
            number_channels_valid=len(record.leads),
            number_channels_allocated=len(record.leads),
        )

        mains = _number(data["raw"].get("mainsFrequency"))
        if mains is not None:
            record.annotations["mains_frequency_hz"] = f"{mains:g}"

        record.measurements = self._read_measurements(doc, record.annotations)

        determination = _text(doc.get("algorithmDetermination"))
        if determination:
            record.interpretation = Interpretation(
                statements=[(determination, "")], source="machine")

        if _text(doc.get("id")):
            record.annotations["recording_id"] = _text(doc.get("id"))
        if _text(doc.get("note")):
            record.annotations["note"] = _text(doc.get("note"))
        tags = doc.get("tags")
        if isinstance(tags, list) and tags:
            record.annotations["tags"] = ", ".join(_text(t) for t in tags if _text(t))

        fill_signal_summary(record)
        return record

    def _read_block(self, block: dict, name: str, file_path: Path) -> tuple[list[Lead], dict]:
        rate = _number(block.get("frequency"))
        if rate is None or rate <= 0:
            raise CorruptedFileError(
                f"{file_path}: data.{name}.frequency is missing or invalid "
                f"({block.get('frequency')!r})"
            )
        samples = block.get("samples")
        if not isinstance(samples, dict) or not samples:
            raise CorruptedFileError(f"{file_path}: no samples in data.{name}")

        resolution_nv = _number(block.get("amplitudeResolution"))
        if resolution_nv is None or resolution_nv <= 0:
            warnings.warn(
                f"Kardia data.{name} has no amplitudeResolution, the samples are left "
                "as raw counts",
                stacklevel=3,
            )
            resolution, unit, adc_res, adc_unit = 1.0, "", 0.0, ""
        else:
            # nV per count, kept in uV per count like the other parsers
            resolution, unit, adc_res, adc_unit = resolution_nv / 1000.0, "uV", resolution_nv, "nV"
        is_raw = derive_is_raw(resolution, 0.0, unit)

        labels = unique_labels([_LEAD_NAMES.get(k, normalize_lead_label(k)) for k in samples])
        leads = []
        for label, values in zip(labels, samples.values()):
            if not isinstance(values, list):
                raise CorruptedFileError(f"{file_path}: data.{name} lead {label} is not a list")
            array = np.array(values, dtype=np.float64)
            if array.ndim != 1:
                raise CorruptedFileError(f"{file_path}: data.{name} lead {label} is not a list of numbers")
            if not np.isfinite(array).all():
                warnings.warn(f"Kardia data.{name} lead {label} has missing samples (NaN)",
                              stacklevel=3)
            leads.append(Lead(
                label=label,
                samples=array,
                sampling_rate=int(round(rate)),
                resolution=resolution,
                resolution_unit=unit,
                units="" if is_raw else unit,
                is_raw=is_raw,
                adc_resolution=adc_res,
                adc_resolution_unit=adc_unit,
            ))

        info = {k: v for k, v in block.items() if k != "samples"}
        num_leads = _number(block.get("numLeads"))
        if num_leads is not None and int(num_leads) != len(leads):
            warnings.warn(
                f"Kardia data.{name}.numLeads is {int(num_leads)} but {len(leads)} "
                "leads are stored",
                stacklevel=3,
            )
        return leads, info

    @staticmethod
    def _read_measurements(doc: dict, annotations: dict[str, str]) -> GlobalMeasurements:
        m = GlobalMeasurements()
        heart_rate = _number(doc.get("heartRate"))
        if heart_rate is not None:
            m.heart_rate = int(round(heart_rate))
        qt = doc.get("qtMeasurement")
        if not isinstance(qt, dict):
            return m
        for key, entry in qt.items():
            value = entry.get("Value") if isinstance(entry, dict) else entry
            field = _QT_FIELDS.get(key.lower())
            number = _number(value)
            if field and number is not None:
                setattr(m, field, int(round(number)))
            elif _text(value):
                annotations[f"qt_measurement_{key}"] = _text(value)
        for key in ("source", "sourceProviderID"):
            if _text(doc.get(key)):
                annotations[f"qt_{key}"] = _text(doc.get(key))
        return m

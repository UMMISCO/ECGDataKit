"""XLE ECG parser (experimental).

There is no public description of this format. The layout below was
worked out from sample files and has not been confirmed by a
specification or by the recorder's own software.

* ``.xle`` files with a 256 KiB (``0x40000`` byte) header followed by the
  signal up to the end of the file.
* Header: patient fields stored as text with the two bytes of each 16-bit
  word swapped, each value after its label (``Given name:``,
  ``Family name:``, ``ID:``, ``Birth Date:``, ``Sex:``, ``History:``,
  ``Comment:``) at fixed offsets and ended by a NUL. The other header
  bytes are not decoded.
* Signal: 8 channels interleaved, signed 16-bit little-endian, 16 bytes
  per sample instant. Only the low 16 bits of each value are stored, so a
  value that drifts past +32767 continues from -32768: consecutive
  differences are taken modulo 65536 to rebuild a continuous signal, the
  first stored value being the origin.

Not stored in the file, and therefore not filled in:

* the voltage scale: samples are raw ADC counts (``is_raw=True``, no
  unit), never converted;
* the lead labels: channels are named ``Ch1`` to ``Ch8``. In the sample
  files their waveforms match I, II, V1 to V6 in that order, recorded in
  ``raw_metadata["likely_lead_order"]`` but not applied;
* the recording date.

The sampling rate is not stored either. It is set to 1000 Hz, the rate
that matches the sample files (heart cycle intervals, and mains
interference at 50 Hz and its harmonics), and
``raw_metadata["sampling_rate_stated"]`` is ``False``.
"""

from __future__ import annotations

import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np

from ecgdatakit.exceptions import CorruptedFileError
from ecgdatakit.models import ECGRecord, Lead, PatientInfo, SignalCharacteristics
from ecgdatakit.parsing.helpers import fill_signal_summary
from ecgdatakit.parsing.parser import Parser

_EXT = ".xle"
_HEADER_SIZE = 0x40000
_CHANNELS = 8
_FRAME = _CHANNELS * 2
_SAMPLING_RATE = 1000
_LIKELY_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]

# Offset of each label in the byte-swapped header, and the field it names
_FIELDS = (
    ("Given name", 2673),
    ("Family name", 2712),
    ("ID", 2761),
    ("Birth Date", 2793),
    ("Sex", 2840),
    ("History", 2876),
    ("Comment", 2916),
)
_TEXT_END = 3200
_SEX = {"M": "M", "F": "F", "W": "F"}  # W: weiblich


def _swap16(data: bytes) -> bytes:
    """*data* with the two bytes of every 16-bit word exchanged."""
    out = bytearray(data[: len(data) // 2 * 2])
    out[0::2], out[1::2] = data[1:len(out):2], data[0:len(out):2]
    return bytes(out)


def _has_labels(header: bytes) -> bool:
    text = _swap16(header[:_TEXT_END])
    return all(
        text[off:off + len(label) + 1] == label.encode("ascii") + b":"
        for label, off in _FIELDS
    )


def _fields(header: bytes) -> dict[str, str]:
    """Value of every patient field, decoded up to its NUL."""
    text = _swap16(header[:_TEXT_END])
    values: dict[str, str] = {}
    for k, (label, off) in enumerate(_FIELDS):
        start = off + len(label) + 1
        end = _FIELDS[k + 1][1] if k + 1 < len(_FIELDS) else _TEXT_END
        raw = text[start:end].split(b"\x00", 1)[0]
        values[label] = raw.decode("latin-1").strip()
    return values


def _unwrap(stored: np.ndarray) -> tuple[np.ndarray, int]:
    """Continuous signal from 16-bit values stored modulo 65536, and the
    number of wraps undone."""
    values = stored.astype(np.int64)
    diff = np.diff(values, axis=0)
    wrapped = (diff + 32768) % 65536 - 32768
    wraps = int(np.count_nonzero(wrapped != diff))
    out = np.empty_like(values)
    out[:1] = values[:1]
    np.cumsum(wrapped, axis=0, out=out[1:])
    out[1:] += values[:1]
    return out.astype(np.float64), wraps


class XLEParser(Parser):
    """Parser for ``.xle`` ECG files (experimental, reverse-engineered)."""

    FORMAT_NAME = "XLE ECG"
    FORMAT_DESCRIPTION = "XLE 8-channel ECG (.xle), experimental"
    FILE_EXTENSIONS = [_EXT]
    PRIORITY = 45

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        path = Path(file_path)
        if path.suffix.lower() != _EXT or not path.is_file():
            return False
        size = path.stat().st_size
        if size <= _HEADER_SIZE or (size - _HEADER_SIZE) % _FRAME:
            return False
        return len(header) >= _TEXT_END and _has_labels(header)

    def parse(self, file_path: Path) -> ECGRecord:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        warnings.warn(
            "XLE support is experimental: the format is not documented. The "
            "layout, the 1000 Hz sampling rate and the channel order were "
            "worked out from sample files, and the voltage scale is unknown, "
            "so samples are raw ADC counts. Check them against the recorder's "
            "own export before use.",
            UserWarning,
            stacklevel=2,
        )
        data = path.read_bytes()
        if len(data) <= _HEADER_SIZE or not _has_labels(data[:_TEXT_END]):
            raise CorruptedFileError(f"Not an XLE ECG file: {path}")
        frames = (len(data) - _HEADER_SIZE) // _FRAME
        leftover = len(data) - _HEADER_SIZE - frames * _FRAME
        stored = np.frombuffer(data, dtype="<i2", count=frames * _CHANNELS,
                               offset=_HEADER_SIZE).reshape(frames, _CHANNELS)
        samples, wraps = _unwrap(stored)

        fields = _fields(data)
        patient = PatientInfo(
            patient_id=fields["ID"],
            first_name=fields["Given name"],
            last_name=fields["Family name"],
            clinical_history=fields["History"],
        )
        sex = fields["Sex"]
        if sex:
            patient.sex = _SEX.get(sex[0].upper(), "U")

        record = ECGRecord(source_format="xle")
        record.patient = patient
        record.recording.duration = timedelta(seconds=frames / _SAMPLING_RATE)
        for ch in range(_CHANNELS):
            record.leads.append(Lead(
                label=f"Ch{ch + 1}",
                samples=np.ascontiguousarray(samples[:, ch]),
                sampling_rate=_SAMPLING_RATE,
            ))
        record.recording.acquisition.signal = SignalCharacteristics(
            sampling_rate=_SAMPLING_RATE,
            bits_per_sample=16,
            signal_signed=True,
            number_channels_allocated=_CHANNELS,
            number_channels_valid=_CHANNELS,
            data_encoding="int16",
            compression="none",
        )

        meta = record.raw_metadata
        meta["filepath"] = str(path)
        meta["reverse_engineered"] = True
        meta["sampling_rate_stated"] = False
        meta["likely_lead_order"] = list(_LIKELY_LEADS)
        meta["header_size"] = _HEADER_SIZE
        meta["samples_per_channel"] = frames
        meta["wraps_undone"] = wraps
        meta["birth_date_text"] = fields["Birth Date"]
        meta["sex_text"] = sex
        meta["comment"] = fields["Comment"]
        if leftover:
            warnings.warn(f"XLE file ends with a partial frame ({leftover} bytes); ignored.",
                          UserWarning, stacklevel=2)
        fill_signal_summary(record)
        return record

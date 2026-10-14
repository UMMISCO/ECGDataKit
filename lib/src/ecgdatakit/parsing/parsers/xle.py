"""XLE ECG parser (experimental).

The manufacturer gives no public description of this format. The layout
below was worked out from sample files and a third-party reader of the
format, and has not been confirmed by a specification or by the
recorder's own software.

* ``.xle`` files with a 256 KiB (``0x40000`` byte) header followed by the
  signal up to the end of the file.
* Header: patient fields stored as text with the two bytes of each 16-bit
  word swapped, each value after its label (``Given name:``,
  ``Family name:``, ``ID:``, ``Birth Date:``, ``Sex:``, ``History:``,
  ``Comment:``) at fixed offsets and ended by a NUL. The recording date
  (``DD.MM.YY`` or ``MM/DD/YY``) and time (``HH:MM``) are text swapped the
  same way, at fixed offsets. The other header bytes are not decoded.
* Signal: 8 channels interleaved, signed 16-bit little-endian, 16 bytes
  per sample instant. Only the low 16 bits of each value are stored, so a
  value that drifts past +32767 continues from -32768: consecutive
  differences are taken modulo 65536 to rebuild a continuous signal, the
  first stored value being the origin. The channels are I, II, then the
  six chest leads measured against the right arm (``CR1`` to ``CR6``).

Not stored in the file, and taken from outside it:

* the voltage scale: 0.9375 µV per count (15/16), from the third-party
  reader and not from the manufacturer's documentation. Samples stay raw
  counts with this resolution, so ``auto_scale`` converts them. The
  conversion warns that the scale is not validated
  (``raw_metadata["scale_warning"]``);
* the sampling rate: 1000 Hz, the rate that matches the sample files
  (heart cycle intervals, and mains interference at 50 Hz and its
  harmonics). ``raw_metadata["sampling_rate_stated"]`` is ``False``.

``XLEParser(wilson_correction=True)`` refers the chest leads to Wilson's
central terminal, ``V = CR - (I + II) / 3``, and names them ``V1`` to
``V6``, with a warning. Leads III, aVR, aVL and aVF are not stored and not
computed.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
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
_RESOLUTION_UV = 15 / 16  # not validated by the manufacturer's documentation
_SCALE_WARNING = (
    "XLE: the voltage scale is not stored in the file. The samples were "
    "converted with 0.9375 uV per count, a scale from a third-party reader "
    "that is not validated by the manufacturer's documentation. Converted "
    "amplitudes are experimental: check them against the recorder's own "
    "printout before use."
)

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
# Recording date (8 characters) and time (5 characters), byte-swapped
_DATE_OFFSET = 182972
_TIME_OFFSET = 182982


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


def _recording_date(data: bytes) -> tuple[datetime | None, str, str]:
    """Recording start, and the date and time text it was read from.
    Dates with dots are day first, dates with slashes month first."""
    date = _swap16(data[_DATE_OFFSET:_DATE_OFFSET + 8]).split(b"\x00", 1)[0]
    time = _swap16(data[_TIME_OFFSET:_TIME_OFFSET + 6])[:5].split(b"\x00", 1)[0]
    date_text = date.decode("latin-1").strip()
    time_text = time.decode("latin-1").strip()
    fmt = "%d.%m.%y" if "." in date_text else "%m/%d/%y"
    try:
        start = datetime.strptime(date_text, fmt)
    except ValueError:
        return None, date_text, time_text
    try:
        clock = datetime.strptime(time_text, "%H:%M")
        start = start.replace(hour=clock.hour, minute=clock.minute)
    except ValueError:
        pass
    return start, date_text, time_text


class XLEParser(Parser):
    """Parser for ``.xle`` ECG files (experimental, reverse-engineered)."""

    FORMAT_NAME = "XLE ECG"
    FORMAT_DESCRIPTION = "XLE 8-channel ECG (.xle), experimental"
    FILE_EXTENSIONS = [_EXT]
    PRIORITY = 45

    def __init__(self, wilson_correction: bool = False) -> None:
        """*wilson_correction*: refer the chest leads, stored against the
        right arm, to Wilson's central terminal (experimental)."""
        self.wilson_correction = wilson_correction

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
            "XLE support is experimental: the format is not documented by its "
            "manufacturer. The layout, the 1000 Hz sampling rate and the lead "
            "order were worked out from sample files and a third-party reader, "
            "and are not validated by the manufacturer's documentation.",
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
        chest = "CR"
        if self.wilson_correction:
            warnings.warn(
                "Wilson correction applied to the XLE chest leads, "
                "V = CR - (I + II) / 3: they are stored against the right arm. "
                "This correction is experimental and not validated by the "
                "manufacturer's documentation.",
                UserWarning,
                stacklevel=2,
            )
            samples[:, 2:] -= (samples[:, :1] + samples[:, 1:2]) / 3
            chest = "V"
        labels = ["I", "II"] + [f"{chest}{k}" for k in range(1, 7)]

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
        start, date_text, time_text = _recording_date(data)
        record.recording.date = start
        record.recording.duration = timedelta(seconds=frames / _SAMPLING_RATE)
        for ch in range(_CHANNELS):
            record.leads.append(Lead(
                label=labels[ch],
                samples=np.ascontiguousarray(samples[:, ch]),
                sampling_rate=_SAMPLING_RATE,
                resolution=_RESOLUTION_UV,
                resolution_unit="uV",
                is_raw=True,
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
        meta["resolution_stated"] = False
        meta["scale_warning"] = _SCALE_WARNING
        meta["lead_order_stated"] = False
        meta["wilson_correction"] = self.wilson_correction
        meta["chest_lead_reference"] = (
            "Wilson central terminal" if self.wilson_correction else "right arm")
        meta["header_size"] = _HEADER_SIZE
        meta["samples_per_channel"] = frames
        meta["wraps_undone"] = wraps
        meta["recording_date_text"] = date_text
        meta["recording_time_text"] = time_text
        meta["birth_date_text"] = fields["Birth Date"]
        meta["sex_text"] = sex
        meta["comment"] = fields["Comment"]
        if leftover:
            warnings.warn(f"XLE file ends with a partial frame ({leftover} bytes); ignored.",
                          UserWarning, stacklevel=2)
        fill_signal_summary(record)
        return record

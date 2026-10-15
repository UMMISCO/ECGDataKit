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
  same way, at fixed offsets.
* Header, representative beat: the QRS onset and offset of the beat (16-bit,
  in ms then in beat samples), the number of QRS complexes detected and
  averaged (32-bit), then 5000 rows of nine 32-bit values (each stored as two
  16-bit little-endian words, high word first). A row holds, on the 8
  channels, the change from the previous instant of the sum of the averaged
  beats, then a per-sample count whose meaning is not confirmed. The count
  starts at the number of beats averaged, and on the sample files it falls
  to 0 or below where the beat data ends. The other header bytes are not
  decoded.
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

The averaged beat goes to ``median_beats``: the stored changes are added up
and divided by the number of beats averaged, up to the first row whose count
is 0 or below, at 5000 Hz (the QRS bounds stored in ms and in beat samples differ by
a factor of 5), with the channels in the order of the leads. Its voltage
scale is not stored either: it is set to the scale of the leads divided by
1.6, the ratio between the beat and the median beat of the leads in counts
on the sample files (it would be 8/5 if each 1000 Hz sample were the sum of
five 5000 Hz samples divided by 8). The beat stays in raw counts with this
resolution, so ``auto_scale`` converts it, with the scale warning. Its QRS
bounds go to ``annotations`` (``median_qrs_onset_ms``,
``median_qrs_offset_ms``, with ``median_beats_averaged``) and the number of
QRS complexes detected to ``measurements.qrs_count``. Reading it issues a
warning.

``XLEParser(wilson_correction=True)`` refers the chest leads to Wilson's
central terminal, ``V = CR - (I + II) / 3``, and names them ``V1`` to
``V6``, with a warning. Leads III, aVR, aVL and aVF are not stored.
``XLEParser(wilson_correction=True, compute_missing_leads=True)`` also
computes them from I and II (Einthoven and Goldberger), with a warning,
so the record holds the standard 12 leads. ``compute_missing_leads``
requires ``wilson_correction``. Both options apply to the representative
beat as well.
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
# Beat counts per lead count, estimated from sample files (not stored)
_AVERAGE_GAIN = 1.6
_AVERAGE_SCALE_WARNING = (
    " The representative beat was converted with 0.9375 / 1.6 uV per count: "
    "the factor 1.6 is the ratio between the beat and the median beat of the "
    "leads on sample files, not a value stored in the file."
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
_TEXT_END = 0xB88  # end of the Comment field: the representative beat follows
_SEX = {"M": "M", "F": "F", "W": "F"}  # W: weiblich
# Recording date (8 characters) and time (5 characters), byte-swapped
_DATE_OFFSET = 182972
_TIME_OFFSET = 182982
# Representative beat
_QRS_OFFSET = 0xB8A  # QRS onset, offset (ms), then the same in beat samples
_COUNTS_OFFSET = 0xB94  # QRS complexes detected, then averaged
_AVERAGE_OFFSET = 0xB9C
_AVERAGE_ROWS = 5000
_AVERAGE_COLUMNS = 9  # 8 channels, then a count whose meaning is not confirmed
_AVERAGE_RATE = 5000
_AVERAGE_WARNING = (
    "XLE representative beat (experimental): the averaged beat stored in the "
    "header, rebuilt from its stored sample-to-sample changes and divided by "
    "the number of beats averaged. Its 5000 Hz sampling rate and its lead "
    "order are inferred, and its voltage scale, not stored, is estimated from "
    "sample files. Not validated by the manufacturer's documentation."
)


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


def _int32(data: bytes, offset: int, count: int) -> np.ndarray:
    """*count* 32-bit values stored as two 16-bit words, high word first."""
    words = np.frombuffer(data, dtype="<i2", count=2 * count, offset=offset).astype(np.int64)
    return words[0::2] * 65536 + (words[1::2] & 0xFFFF)


def _average(data: bytes) -> tuple[np.ndarray, dict[str, int]] | None:
    """Averaged beat (samples x 8) and the values stored with it, or
    ``None`` when it is empty."""
    table = _int32(data, _AVERAGE_OFFSET, _AVERAGE_ROWS * _AVERAGE_COLUMNS)
    table = table.reshape(_AVERAGE_ROWS, _AVERAGE_COLUMNS)
    counts = table[:, -1]
    onset, offset = np.frombuffer(data, dtype="<u2", count=2, offset=_QRS_OFFSET)
    detected, averaged = _int32(data, _COUNTS_OFFSET, 2)
    # on the sample files the beat data ends where the count falls to 0 or
    # below, and the rows after it hold zeros or unrelated values
    ended = np.flatnonzero(counts <= 0)
    end = int(ended[0]) if ended.size else _AVERAGE_ROWS
    if end == 0 or averaged <= 0:
        return None
    beat = np.cumsum(table[:end, :-1], axis=0) / averaged
    return beat, {"qrs_onset_ms": int(onset), "qrs_offset_ms": int(offset),
                  "detected": int(detected), "averaged": int(averaged)}


def _lead_set(samples: np.ndarray, wilson: bool, missing: bool) -> list[tuple[str, np.ndarray]]:
    """Labelled leads from *samples* (I, II, then the chest leads against the
    right arm), modified in place by the Wilson correction."""
    chest = "CR"
    if wilson:
        samples[:, 2:] -= (samples[:, :1] + samples[:, 1:2]) / 3
        chest = "V"
    lead_i, lead_ii = samples[:, 0], samples[:, 1]
    leads = [("I", lead_i), ("II", lead_ii)]
    if missing:
        leads += [
            ("III", lead_ii - lead_i),
            ("aVR", -(lead_i + lead_ii) / 2),
            ("aVL", lead_i - lead_ii / 2),
            ("aVF", lead_ii - lead_i / 2),
        ]
    return leads + [(f"{chest}{k}", samples[:, k + 1]) for k in range(1, 7)]


class XLEParser(Parser):
    """Parser for ``.xle`` ECG files (experimental, reverse-engineered)."""

    FORMAT_NAME = "XLE ECG"
    FORMAT_DESCRIPTION = "XLE 8-channel ECG (.xle), experimental"
    FILE_EXTENSIONS = [_EXT]
    PRIORITY = 45

    def __init__(self, wilson_correction: bool = False,
                 compute_missing_leads: bool = False) -> None:
        """*wilson_correction*: refer the chest leads, stored against the
        right arm, to Wilson's central terminal (experimental).
        *compute_missing_leads*: add III, aVR, aVL and aVF, computed from
        I and II. Requires *wilson_correction*, so that the record holds
        the standard 12 leads.

        Raises
        ------
        ValueError
            If *compute_missing_leads* is set without *wilson_correction*.
        """
        if compute_missing_leads and not wilson_correction:
            raise ValueError(
                "compute_missing_leads=True requires wilson_correction=True: "
                "the 12 standard leads need the chest leads referred to "
                "Wilson's central terminal."
            )
        self.wilson_correction = wilson_correction
        self.compute_missing_leads = compute_missing_leads

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
        if self.wilson_correction:
            warnings.warn(
                "Wilson correction applied to the XLE chest leads, "
                "V = CR - (I + II) / 3: they are stored against the right arm. "
                "This correction is experimental and not validated by the "
                "manufacturer's documentation.",
                UserWarning,
                stacklevel=2,
            )
        computed: list[str] = []
        if self.compute_missing_leads:
            warnings.warn(
                "Leads III, aVR, aVL and aVF are not stored in the XLE file: "
                "they were computed from leads I and II.",
                UserWarning,
                stacklevel=2,
            )
            computed = ["III", "aVR", "aVL", "aVF"]
        leads = _lead_set(samples, self.wilson_correction, self.compute_missing_leads)
        average = _average(data)

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
        for label, values in leads:
            record.leads.append(Lead(
                label=label,
                samples=np.ascontiguousarray(values),
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
        if average is not None:
            warnings.warn(_AVERAGE_WARNING, UserWarning, stacklevel=2)
            beat, stored_values = average
            for label, values in _lead_set(beat, self.wilson_correction,
                                           self.compute_missing_leads):
                record.median_beats.append(Lead(
                    label=label,
                    samples=np.ascontiguousarray(values),
                    sampling_rate=_AVERAGE_RATE,
                    resolution=_RESOLUTION_UV / _AVERAGE_GAIN,
                    resolution_unit="uV",
                    is_raw=True,
                ))
            record.annotations["median_qrs_onset_ms"] = str(stored_values["qrs_onset_ms"])
            record.annotations["median_qrs_offset_ms"] = str(stored_values["qrs_offset_ms"])
            record.annotations["median_beats_averaged"] = str(stored_values["averaged"])
            record.measurements.qrs_count = stored_values["detected"]

        meta = record.raw_metadata
        meta["filepath"] = str(path)
        meta["reverse_engineered"] = True
        meta["sampling_rate_stated"] = False
        meta["resolution_stated"] = False
        meta["scale_warning"] = _SCALE_WARNING + (
            _AVERAGE_SCALE_WARNING if average is not None else "")
        meta["lead_order_stated"] = False
        meta["wilson_correction"] = self.wilson_correction
        meta["chest_lead_reference"] = (
            "Wilson central terminal" if self.wilson_correction else "right arm")
        meta["computed_leads"] = computed
        if average is not None:
            meta["median_sampling_rate_stated"] = False
            meta["median_scale_stated"] = False
            meta["median_scale_factor"] = _AVERAGE_GAIN
            meta["median_lead_order_stated"] = False
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

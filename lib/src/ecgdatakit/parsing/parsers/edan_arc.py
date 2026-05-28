"""EDAN ARC / SE-2012 Holter format parser.

Reference: https://paulbourke.net/dataformats/edan/

The reference recording layout is a pair of companion files:

* ``patient.hea`` — fixed-offset binary header (~3 kB) carrying patient
  demographics, device info, channel count, sampling rate and per-channel
  electrode labels.
* ``ecgraw.dat`` — raw signal as little-endian ``uint16`` samples,
  interleaved by channel.  ADC zero is 16384, so signed-centred values
  are obtained via ``s - 16384``.

EDAN's analysis software can also export a single ``.arc`` archive that
concatenates the above files behind a wrapper.  The wrapper format is
**undocumented**; we support ``.arc`` on a *best-effort* basis by
scanning for an embedded ``patient.hea`` and ``ecgraw.dat`` using the
documented header fingerprint, and emit a :class:`UserWarning` when this
path is taken.
"""

from __future__ import annotations

import struct
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ecgdatakit.exceptions import CorruptedFileError
from ecgdatakit.models import (
    DeviceInfo,
    ECGRecord,
    FilterSettings,
    Lead,
    PatientInfo,
    RecordingInfo,
    SignalCharacteristics,
)
from ecgdatakit.parsing.parser import Parser

_HEA_NAME = "patient.hea"
_DAT_NAME = "ecgraw.dat"
_ARC_EXT = ".arc"
_ADC_ZERO = 16384
_MAX_CHANNELS = 12
_HEA_BLOB_MAX = 4096
# Plausible Unix epochs: 1990-01-01 .. 2070-01-01
_EPOCH_MIN = 631_152_000
_EPOCH_MAX = 3_155_760_000


def _ascii(buf: bytes, offset: int, length: int) -> str:
    """Decode an ASCII slice, stripping NULs and surrounding whitespace."""
    if offset < 0 or offset + length > len(buf):
        return ""
    raw = buf[offset:offset + length]
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()


def _s16(buf: bytes, offset: int) -> int:
    """Read a little-endian signed 16-bit integer (0 if out of range)."""
    if offset + 2 > len(buf) or offset < 0:
        return 0
    return struct.unpack_from("<h", buf, offset)[0]


def _u32(buf: bytes, offset: int) -> int:
    """Read a little-endian unsigned 32-bit integer (0 if out of range)."""
    if offset + 4 > len(buf) or offset < 0:
        return 0
    return struct.unpack_from("<I", buf, offset)[0]


def _epoch_to_datetime(seconds: int) -> datetime | None:
    """Convert a Unix-epoch second count to a naive UTC datetime."""
    if seconds <= 0:
        return None
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)
    except (OverflowError, OSError, ValueError):
        return None


def _looks_like_patient_hea(buf: bytes, off: int) -> bool:
    """Structural fingerprint for a patient.hea header at offset *off*.

    Validates three independent fields against plausible ranges:
    channel count (byte 12), sampling rate (s16 at 32), and the start
    timestamp (u32 at 4).  Combined the false-positive rate is very low.
    """
    if off < 0 or off + 64 > len(buf):
        return False
    nch = buf[off + 12]
    if not (1 <= nch <= _MAX_CHANNELS):
        return False
    sr = struct.unpack_from("<h", buf, off + 32)[0]
    if not (50 <= sr <= 4000):
        return False
    epoch = struct.unpack_from("<I", buf, off + 4)[0]
    if not (_EPOCH_MIN <= epoch <= _EPOCH_MAX):
        return False
    return True


def _scan_for_patient_hea(arc: bytes) -> int:
    """Locate the embedded patient.hea header in an .arc archive.

    Tries the ``patient.hea`` ASCII filename marker first (probing a
    short window of post-marker offsets), then falls back to a
    structural fingerprint scan over the whole file.
    """
    marker = _HEA_NAME.encode("ascii")
    search_start = 0
    while True:
        pos = arc.find(marker, search_start)
        if pos == -1:
            break
        # Wrapper metadata after the filename literal is unknown; probe a
        # plausible window for the start of the actual header.
        for delta in range(0, 256):
            start = pos + len(marker) + delta
            if _looks_like_patient_hea(arc, start):
                return start
        search_start = pos + len(marker)

    # Pure fingerprint scan (4-byte aligned to keep it cheap).
    for off in range(0, max(0, len(arc) - 64), 4):
        if _looks_like_patient_hea(arc, off):
            return off

    raise CorruptedFileError(
        "Could not locate an embedded patient.hea header inside .arc "
        "(EDAN .arc wrapper format is undocumented)."
    )


def _scan_for_ecgraw_dat(
    arc: bytes,
    channel_count: int,
    hea_end: int,
) -> bytes:
    """Locate the embedded ecgraw.dat payload in an .arc archive.

    Strategy:
    1. Look for the ``ecgraw.dat`` ASCII filename marker; the bytes
       following the wrapper metadata are taken as the payload.
    2. If no marker is found, take everything after the patient.hea
       header as the payload — for EDAN exports ecgraw.dat is by far
       the largest section, so this is a reasonable fallback.
    """
    marker = _DAT_NAME.encode("ascii")
    pos = arc.find(marker)
    if pos != -1:
        for delta in range(0, 256):
            start = pos + len(marker) + delta
            if start >= len(arc):
                break
            payload = arc[start:]
            # Trim to integer multiple of 2 * channel_count
            stride = 2 * channel_count
            usable = (len(payload) // stride) * stride
            if usable >= stride:  # at least one full time-step
                return payload[:usable]

    payload = arc[hea_end:]
    stride = 2 * channel_count
    usable = (len(payload) // stride) * stride
    if usable < stride:
        raise CorruptedFileError(
            "No usable ecgraw.dat payload found in .arc after patient.hea"
        )
    return payload[:usable]


class EDANARCHolterParser(Parser):
    """Parser for EDAN SE-2012 / ARC Holter recordings.

    Accepts two entry points:

    * ``patient.hea`` next to ``ecgraw.dat`` — the documented layout
      (fully deterministic).
    * ``*.arc`` — best-effort heuristic extraction (emits a warning).
    """

    FORMAT_NAME = "EDAN ARC Holter"
    FORMAT_DESCRIPTION = "EDAN SE-2012 / ARC Holter (patient.hea + ecgraw.dat or .arc)"
    FILE_EXTENSIONS = [".hea", ".dat", ".arc"]

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        name_lower = file_path.name.lower()
        if name_lower.endswith(_ARC_EXT):
            return True
        if name_lower != _HEA_NAME:
            return False
        dat = file_path.parent / _DAT_NAME
        if dat.exists():
            return True
        try:
            siblings = {p.name.lower(): p for p in file_path.parent.iterdir()}
        except OSError:
            return False
        return _DAT_NAME in siblings

    def parse(self, file_path: Path) -> ECGRecord:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.name.lower().endswith(_ARC_EXT):
            hea_bytes, dat_bytes = self._read_arc(path)
            source = "edan_arc_archive"
            arc_filepath: str | None = str(path)
            hea_filepath: str | None = None
            dat_filepath: str | None = None
        else:
            hea_bytes, dat_bytes, dat_path = self._read_companion_files(path)
            source = "edan_arc"
            arc_filepath = None
            hea_filepath = str(path)
            dat_filepath = str(dat_path)

        record = self._build_record(hea_bytes, dat_bytes, source)
        record.raw_metadata["filepath"] = arc_filepath or hea_filepath
        if hea_filepath:
            record.raw_metadata["hea_filepath"] = hea_filepath
        if dat_filepath:
            record.raw_metadata["data_filepath"] = dat_filepath
        if arc_filepath:
            record.raw_metadata["arc_filepath"] = arc_filepath
            record.raw_metadata["arc_heuristic"] = True
        return record

    # I/O helpers

    @staticmethod
    def _read_companion_files(hea_path: Path) -> tuple[bytes, bytes, Path]:
        dat_path = hea_path.parent / _DAT_NAME
        if not dat_path.exists():
            siblings = {p.name.lower(): p for p in hea_path.parent.iterdir()}
            dat_path = siblings.get(_DAT_NAME, dat_path)
        if not dat_path.exists():
            raise CorruptedFileError(
                f"Companion {_DAT_NAME} not found next to {hea_path.name}"
            )
        hea_bytes = hea_path.read_bytes()
        dat_bytes = dat_path.read_bytes()
        return hea_bytes, dat_bytes, dat_path

    @staticmethod
    def _read_arc(arc_path: Path) -> tuple[bytes, bytes]:
        warnings.warn(
            "Parsing EDAN .arc archives is best-effort: the wrapper format "
            "is undocumented. Sample values and metadata should be sanity-"
            "checked against the recorder's own export.",
            UserWarning,
            stacklevel=3,
        )
        arc = arc_path.read_bytes()
        if len(arc) < 64:
            raise CorruptedFileError(
                f"EDAN .arc too small: {len(arc)} bytes"
            )
        hea_off = _scan_for_patient_hea(arc)
        hea_bytes = arc[hea_off:hea_off + _HEA_BLOB_MAX]
        # Peek channel count to size the data scan
        channel_count = hea_bytes[12] if len(hea_bytes) > 12 else 0
        if not (1 <= channel_count <= _MAX_CHANNELS):
            raise CorruptedFileError(
                f"Embedded patient.hea has invalid channel count: {channel_count}"
            )
        hea_end = hea_off + _HEA_BLOB_MAX
        dat_bytes = _scan_for_ecgraw_dat(arc, channel_count, hea_end)
        return hea_bytes, dat_bytes

    # Core decode

    @staticmethod
    def _build_record(hea: bytes, dat: bytes, source_format: str) -> ECGRecord:
        if len(hea) < 64:
            raise CorruptedFileError(
                f"EDAN patient.hea too small: {len(hea)} bytes"
            )

        record = ECGRecord(source_format=source_format)

        # Timestamps (Unix epoch seconds at offsets 4 and 8)
        start_epoch = _u32(hea, 4)
        end_epoch = _u32(hea, 8)
        start_dt = _epoch_to_datetime(start_epoch)
        end_dt = _epoch_to_datetime(end_epoch)

        # Core signal layout
        channel_count = hea[12] if len(hea) > 12 else 0
        if channel_count <= 0 or channel_count > _MAX_CHANNELS:
            raise CorruptedFileError(
                f"Invalid EDAN channel count: {channel_count}"
            )

        sampling_rate = _s16(hea, 32)
        if sampling_rate <= 0:
            raise CorruptedFileError(
                f"Invalid EDAN sampling rate: {sampling_rate}"
            )

        height = _s16(hea, 60)
        weight = _s16(hea, 64)
        lowpass = _s16(hea, 2596)

        # Patient demographics
        patient = PatientInfo()
        patient.patient_id = _ascii(hea, 108, 32)
        diagnosis = _ascii(hea, 140, 120)
        medication = _ascii(hea, 242, 102)
        full_name = _ascii(hea, 2637, 64)
        if full_name:
            parts = full_name.split(None, 1)
            patient.first_name = parts[0]
            if len(parts) > 1:
                patient.last_name = parts[1]
        if height > 0:
            patient.height = float(height)
        if weight > 0:
            patient.weight = float(weight)
        if medication:
            patient.medications = [medication]
        if diagnosis:
            patient.clinical_history = diagnosis
        record.patient = patient

        # Recording / device metadata
        recording = RecordingInfo()
        recording.date = start_dt
        recording.end_date = end_dt
        if start_dt and end_dt and end_dt > start_dt:
            recording.duration = end_dt - start_dt
        recording.technician = _ascii(hea, 2764, 64)
        recording.referring_physician = _ascii(hea, 2700, 64)

        device = DeviceInfo(
            manufacturer="EDAN",
            model=_ascii(hea, 2304, 10),
            software_version=_ascii(hea, 2314, 6) or _ascii(hea, 2416, 6),
            department=_ascii(hea, 1960, 134),
        )
        recording.device = device
        record.recording = recording

        # Per-channel electrode labels (8 bytes each from offset 1796)
        lead_labels: list[str] = []
        for ch in range(channel_count):
            label = _ascii(hea, 1796 + ch * 8, 8)
            lead_labels.append(label or f"Ch{ch + 1}")

        # Signal data
        raw = np.frombuffer(dat, dtype="<u2")
        samples_per_channel = raw.size // channel_count
        if samples_per_channel == 0:
            raise CorruptedFileError(
                f"EDAN signal payload contains no samples for "
                f"{channel_count} channels"
            )
        raw = raw[: samples_per_channel * channel_count]
        # Interleaved: 2 x channel_count bytes per time step
        matrix = raw.reshape((samples_per_channel, channel_count))
        signed = matrix.astype(np.float64) - _ADC_ZERO

        for ch in range(channel_count):
            record.leads.append(Lead(
                label=lead_labels[ch],
                samples=np.asarray(signed[:, ch], dtype=np.float64).copy(),
                sampling_rate=sampling_rate,
            ))

        if recording.duration is None:
            recording.duration = timedelta(
                seconds=samples_per_channel / sampling_rate
            )

        # Signal characteristics + filters
        record.recording.acquisition.signal = SignalCharacteristics(
            sampling_rate=sampling_rate,
            bits_per_sample=16,
            signal_offset=_ADC_ZERO,
            signal_signed=False,
            number_channels_allocated=channel_count,
            number_channels_valid=len(record.leads),
            data_encoding="uint16",
            compression="none",
        )
        if lowpass > 0:
            record.recording.acquisition.filters = FilterSettings(lowpass=float(lowpass))

        # Raw metadata
        record.raw_metadata["start_epoch"] = int(start_epoch)
        record.raw_metadata["end_epoch"] = int(end_epoch)
        record.raw_metadata["telephone"] = _ascii(hea, 68, 40)
        record.raw_metadata["accession_number"] = _ascii(hea, 344, 68)
        record.raw_metadata["in_out_pe_id"] = _ascii(hea, 412, 134)
        record.raw_metadata["patient_area"] = _ascii(hea, 546, 134)
        record.raw_metadata["recorder_id"] = _ascii(hea, 2304, 10)
        record.raw_metadata["dft_filter"] = _ascii(hea, 2628, 5)
        record.raw_metadata["procedure"] = _ascii(hea, 2828, 64)
        record.raw_metadata["medical_history"] = _ascii(hea, 2892, 64)
        record.raw_metadata["address"] = _ascii(hea, 2956, 86)
        record.raw_metadata["lead_labels"] = lead_labels

        return record

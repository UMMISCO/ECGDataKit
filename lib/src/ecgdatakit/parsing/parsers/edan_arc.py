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
# EDAN SE-2012 ships with a small set of sampling rates; restricting the
# fingerprint to these dramatically cuts false-positives when scanning a
# multi-MB .arc archive.
_VALID_SAMPLING_RATES = {100, 128, 200, 250, 256, 500, 1000}
# Documented channel counts for SE-2012 (3-lead or 12-lead recorder).
_VALID_CHANNEL_COUNTS = {3, 12}
# Holter recordings rarely exceed a few days — used to validate the
# (start_epoch, end_epoch) pair.
_MAX_RECORDING_SECONDS = 14 * 24 * 3600  # 14 days
# Signature that identifies the "NEUTRAL HOLTER RECORDING" archive
# variant — a *different* EDAN-family format (no public spec) whose
# layout was reverse-engineered from a single sample file:
#
#   offset 0x000  u32         constant 0x00000003 (record count?)
#   offset 0x004  ASCII[28]   "##NEUTRAL HOLTER RECORDING##"
#   offset 0x036  ASCII[~16]  UUID-style session id (e.g. "6a0c3d93-3b15")
#   offset 0x056  u32         file offset to a trailing index section
#   offset 0x05a  ASCII       filename literal "patientdata.dat"
#   offset 0x1000             start of interleaved ECG samples
#                             (3 channels × int16 LE @ 250 Hz)
#   <std-jump>                end of ECG, start of 32-byte beat records
#
# Every field above other than the magic should be treated as
# best-effort: a single-sample reverse-engineering cannot prove field
# semantics, only their byte layout in *this* file.
_NEUTRAL_HOLTER_SIGNATURE = b"##NEUTRAL HOLTER RECORDING##"
_NEUTRAL_HOLTER_SIGNATURE_OFFSET = 4
_NEUTRAL_PAYLOAD_START = 0x1000
_NEUTRAL_CHANNELS = 3
_NEUTRAL_SAMPLING_RATE = 250
_NEUTRAL_UUID_OFFSET = 0x36
_NEUTRAL_UUID_LEN = 20
_NEUTRAL_FILENAME_OFFSET = 0x5A
_NEUTRAL_INDEX_PTR_OFFSET = 0x56
# Threshold separating ECG samples (std ~25) from beat-record bytes
# (std > 1000) when read as int16.
_NEUTRAL_ECG_STD_THRESHOLD = 500.0
_NEUTRAL_SCAN_BLOCK = 4 * 1024


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

    Combines five independent checks against documented SE-2012 values:

    * channel count (byte 12) is 3 or 12,
    * sampling rate (s16 at 32) is one of the SE-2012 rates,
    * both timestamps (u32 at 4 and 8) decode to plausible epochs,
    * the recording length implied by the timestamps is below 14 days,
    * the per-channel lead label slot at offset 1796 contains
      mostly-ASCII bytes.

    The combination yields a very low false-positive rate even when
    scanning a hundred-megabyte .arc archive.
    """
    if off < 0 or off + 64 > len(buf):
        return False
    nch = buf[off + 12]
    if nch not in _VALID_CHANNEL_COUNTS:
        return False
    sr = struct.unpack_from("<h", buf, off + 32)[0]
    if sr not in _VALID_SAMPLING_RATES:
        return False
    start_epoch = struct.unpack_from("<I", buf, off + 4)[0]
    end_epoch = struct.unpack_from("<I", buf, off + 8)[0]
    if not (_EPOCH_MIN <= start_epoch <= _EPOCH_MAX):
        return False
    if not (_EPOCH_MIN <= end_epoch <= _EPOCH_MAX):
        return False
    if not (0 < end_epoch - start_epoch <= _MAX_RECORDING_SECONDS):
        return False
    # Lead labels at offset 1796 are 8-byte ASCII strings — if the file
    # is long enough, sample the first label to confirm it's text.
    label_off = off + 1796
    if label_off + 8 <= len(buf):
        label = buf[label_off:label_off + 8].split(b"\x00", 1)[0]
        if label and not all(32 <= b < 127 for b in label):
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


def _is_neutral_holter(arc: bytes) -> bool:
    """True iff *arc* carries the NEUTRAL HOLTER RECORDING signature.

    The literal must sit at its documented offset (0x04) — accepting any
    location risks colliding with a coincidental in-payload occurrence.
    """
    end = _NEUTRAL_HOLTER_SIGNATURE_OFFSET + len(_NEUTRAL_HOLTER_SIGNATURE)
    if len(arc) < end:
        return False
    return (
        arc[_NEUTRAL_HOLTER_SIGNATURE_OFFSET:end]
        == _NEUTRAL_HOLTER_SIGNATURE
    )


def _find_neutral_holter_payload_end(
    arc: bytes,
    start: int,
    channels: int,
) -> int:
    """Locate the boundary between the ECG samples and the beat-records.

    ECG samples have very low int16 std (~25); the trailing 32-byte beat
    records yield std > 1000.  Scan forward in 16-KB blocks looking for
    the first block whose std crosses the threshold, then refine to a
    channel-aligned offset.
    """
    cursor = start
    block = _NEUTRAL_SCAN_BLOCK
    # Default to "no jump found" → take the rest of the file as signal.
    boundary = len(arc)
    while cursor + block <= len(arc):
        chunk = np.frombuffer(arc[cursor:cursor + block], dtype="<i2")
        if chunk.size and float(chunk.std()) > _NEUTRAL_ECG_STD_THRESHOLD:
            boundary = cursor
            break
        cursor += block

    stride = channels * 2
    return ((boundary - start) // stride) * stride + start


def _parse_filename_timestamp(name: str) -> datetime | None:
    """Best-effort recording-start timestamp from a NEUTRAL HOLTER
    filename like ``DT-06_05_2026-11_38_39.arc``.

    The day/month order isn't documented; this picks the EU convention
    (``DD_MM_YYYY``) because that's what the observed sample uses.
    Returns ``None`` if the pattern doesn't match — the caller treats
    the date as unknown rather than guessing.
    """
    import re
    m = re.search(
        r"(\d{2})_(\d{2})_(\d{4})[-_](\d{2})_(\d{2})_(\d{2})",
        name,
    )
    if m is None:
        return None
    day, month, year, hour, minute, second = (int(g) for g in m.groups())
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


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
        stride = 2 * channel_count
        for delta in range(0, 256):
            start = pos + len(marker) + delta
            if start + 32 > len(arc):
                break
            # Validate: EDAN ecgraw.dat samples are uint16 LE centred at
            # 16384, and the recorder spec hints at +/-256 typical range
            # so plausible values cluster tightly around the mid-scale.
            # Wrapper bytes (NULs, size fields with trailing zeros, etc.)
            # decode to values far from 16384, so a tight ADC range gives
            # a strong fingerprint for the true payload start.
            probe = np.frombuffer(arc[start:start + 32], dtype="<u2")
            if probe.size and np.all((probe > 10_000) & (probe < 24_000)):
                payload = arc[start:]
                usable = (len(payload) // stride) * stride
                if usable >= stride:
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
            arc = path.read_bytes()
            if _is_neutral_holter(arc):
                return self._parse_neutral_holter(path, arc)
            hea_bytes, dat_bytes = self._read_arc(path, arc)
            record = self._build_record(hea_bytes, dat_bytes, "edan_arc_archive")
            record.raw_metadata["filepath"] = str(path)
            record.raw_metadata["arc_filepath"] = str(path)
            record.raw_metadata["arc_heuristic"] = True
            return record

        hea_bytes, dat_bytes, dat_path = self._read_companion_files(path)
        record = self._build_record(hea_bytes, dat_bytes, "edan_arc")
        record.raw_metadata["filepath"] = str(path)
        record.raw_metadata["hea_filepath"] = str(path)
        record.raw_metadata["data_filepath"] = str(dat_path)
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
    def _read_arc(arc_path: Path, arc: bytes) -> tuple[bytes, bytes]:
        warnings.warn(
            "Parsing EDAN .arc archives is best-effort: the wrapper format "
            "is undocumented. Sample values and metadata should be sanity-"
            "checked against the recorder's own export.",
            UserWarning,
            stacklevel=3,
        )
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

    # NEUTRAL HOLTER decode (reverse-engineered, single sample)

    def _parse_neutral_holter(self, arc_path: Path, arc: bytes) -> ECGRecord:
        warnings.warn(
            "NEUTRAL HOLTER RECORDING .arc parsing is reverse-engineered "
            "from a single sample file — no vendor specification exists. "
            "The signal layout (3 channels × int16 LE @ 250 Hz interleaved) "
            "and metadata are best guesses. Validate sample values and "
            "channel order against the recorder's own viewer before any "
            "clinical or research use.",
            UserWarning,
            stacklevel=3,
        )
        if len(arc) < _NEUTRAL_PAYLOAD_START + 4:
            raise CorruptedFileError(
                f"NEUTRAL HOLTER .arc too small: {len(arc)} bytes"
            )

        payload_end = _find_neutral_holter_payload_end(
            arc, _NEUTRAL_PAYLOAD_START, _NEUTRAL_CHANNELS,
        )
        stride = _NEUTRAL_CHANNELS * 2
        payload_size = ((payload_end - _NEUTRAL_PAYLOAD_START) // stride) * stride
        if payload_size < stride:
            raise CorruptedFileError(
                "NEUTRAL HOLTER .arc: no decodable ECG payload found"
            )
        payload = arc[_NEUTRAL_PAYLOAD_START:_NEUTRAL_PAYLOAD_START + payload_size]
        raw = np.frombuffer(payload, dtype="<i2").astype(np.float64)
        matrix = raw.reshape(-1, _NEUTRAL_CHANNELS)
        samples_per_channel = matrix.shape[0]

        record = ECGRecord(source_format="neutral_holter_arc")
        record.recording.duration = timedelta(
            seconds=samples_per_channel / _NEUTRAL_SAMPLING_RATE,
        )
        record.recording.device = DeviceInfo(
            manufacturer="EDAN",
            model="Holter (NEUTRAL HOLTER export)",
        )
        record.recording.acquisition.signal = SignalCharacteristics(
            sampling_rate=_NEUTRAL_SAMPLING_RATE,
            bits_per_sample=16,
            signal_signed=True,
            number_channels_allocated=_NEUTRAL_CHANNELS,
            number_channels_valid=_NEUTRAL_CHANNELS,
            data_encoding="int16",
            compression="none",
        )

        for ch in range(_NEUTRAL_CHANNELS):
            record.leads.append(Lead(
                label=f"Ch{ch + 1}",
                samples=np.asarray(matrix[:, ch], dtype=np.float64).copy(),
                sampling_rate=_NEUTRAL_SAMPLING_RATE,
            ))

        # Best-effort metadata
        session_uuid = _ascii(arc, _NEUTRAL_UUID_OFFSET, _NEUTRAL_UUID_LEN)
        embedded_filename = _ascii(arc, _NEUTRAL_FILENAME_OFFSET, 16)
        index_ptr = _u32(arc, _NEUTRAL_INDEX_PTR_OFFSET)
        # Filename usually carries a recording start timestamp, e.g.
        # "DT-06_05_2026-11_38_39.arc" → 2026-05-06 11:38:39.
        recording_start = _parse_filename_timestamp(arc_path.name)
        if recording_start is not None:
            record.recording.date = recording_start
            if record.recording.duration is not None:
                record.recording.end_date = (
                    recording_start + record.recording.duration
                )

        record.raw_metadata["filepath"] = str(arc_path)
        record.raw_metadata["arc_filepath"] = str(arc_path)
        record.raw_metadata["arc_variant"] = "neutral_holter"
        record.raw_metadata["reverse_engineered"] = True
        record.raw_metadata["session_uuid"] = session_uuid
        record.raw_metadata["embedded_filename"] = embedded_filename
        record.raw_metadata["index_section_offset"] = int(index_ptr)
        record.raw_metadata["ecg_payload_start"] = _NEUTRAL_PAYLOAD_START
        record.raw_metadata["ecg_payload_end"] = (
            _NEUTRAL_PAYLOAD_START + payload_size
        )
        record.raw_metadata["samples_per_channel"] = samples_per_channel
        return record

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

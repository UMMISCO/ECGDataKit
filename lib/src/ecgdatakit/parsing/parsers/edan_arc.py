"""EDAN ARC / SE-2012 Holter format parser.

Reference: https://paulbourke.net/dataformats/edan/ (reverse-engineered,
partial; there is no public vendor specification).

The reference recording layout is a pair of companion files in one
directory:

* ``patient.hea``: fixed-offset binary header (3672 bytes) carrying
  patient demographics, device info, channel count, sampling rate and
  per-channel electrode labels. Text fields are padded with NULs.
* ``ecgraw.dat``: raw signal as little-endian ``uint16`` samples,
  interleaved by channel. ADC zero is 16384, so signed values are
  obtained via ``s - 16384``. The reference states that all 12 channels
  are written even when fewer electrodes are used, while the header
  stores a channel count; the stored channel stride is therefore
  inferred from the data (see :func:`_choose_stride`).

The amplitude scale (uV per count) is not documented, so leads are
returned as raw counts.

EDAN's analysis software can also export a single ``.arc`` archive that
concatenates the above files behind a wrapper. The wrapper format is
**undocumented**; we support ``.arc`` on a *best-effort* basis by
scanning for an embedded ``patient.hea`` and ``ecgraw.dat`` using the
documented header fingerprint, and emit a :class:`UserWarning` when this
path is taken.
"""

from __future__ import annotations

import re
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
from ecgdatakit.parsing.helpers import (
    decode_text,
    fill_signal_summary,
    normalize_lead_label,
    unique_labels,
)
from ecgdatakit.parsing.parser import Parser

_HEA_NAME = "patient.hea"
_DAT_NAME = "ecgraw.dat"
_ARC_EXT = ".arc"
_ADC_ZERO = 16384
_MAX_CHANNELS = 12
# The reference lists "End of file" at offset 3672.
_HEA_SIZE = 3672
_LABEL_OFFSET = 1796
_LABEL_SIZE = 8
# Text fields are documented as ASCII; EDAN is a Chinese vendor, so
# non-ASCII bytes are decoded as GB18030 first (Latin-1 as last resort).
_TEXT_ENCODING = "gb18030"
# Plausible Unix epochs: 1990-01-01 .. 2070-01-01
_EPOCH_MIN = 631_152_000
_EPOCH_MAX = 3_155_760_000
# EDAN SE-2012 sampling rates seen in practice; used for the .arc
# fingerprint and to warn on unusual header values.
_VALID_SAMPLING_RATES = {100, 128, 200, 250, 256, 500, 1000}
_MAX_SAMPLING_RATE = 10_000
# Channel counts accepted by the .arc fingerprint (3-lead or 12-lead).
_VALID_CHANNEL_COUNTS = {3, 12}
# Holter recordings rarely exceed a few days; used by the .arc fingerprint.
_MAX_RECORDING_SECONDS = 14 * 24 * 3600
# Longest header duration trusted when inferring the channel stride.
_MAX_HEADER_SECONDS = 31 * 24 * 3600
# A decoded channel whose median lies further than this from ADC zero is
# not plausible ECG (the reference reports values within about +/-256).
_ADC_PLAUSIBLE_RANGE = 8192
# Frames used to compare candidate channel strides.
_STRIDE_PROBE_FRAMES = 20_000
# Signature that identifies the "NEUTRAL HOLTER RECORDING" archive
# variant, a *different* EDAN-family format (no public spec) whose
# layout was reverse-engineered from a single sample file:
#
#   offset 0x000  u32         constant 0x00000003 (record count?)
#   offset 0x004  ASCII[28]   "##NEUTRAL HOLTER RECORDING##"
#   offset 0x036  ASCII[~16]  UUID-style session id (e.g. "6a0c3d93-3b15")
#   offset 0x056  u32         file offset to a trailing index section
#   offset 0x05a  ASCII       filename literal "patientdata.dat"
#   offset 0x1000             start of interleaved ECG samples
#                             (3 channels x int16 LE @ 250 Hz)
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


def _text(buf: bytes, offset: int, length: int) -> str:
    """Decode a NUL-padded header text field without losing characters.

    Fields cut by a truncated header are decoded up to the end of *buf*.
    """
    if offset < 0 or offset >= len(buf):
        return ""
    raw = buf[offset:offset + length].split(b"\x00", 1)[0]
    return decode_text(raw, _TEXT_ENCODING).strip()


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
    """Convert a plausible Unix-epoch second count to a naive UTC datetime."""
    if not (_EPOCH_MIN <= seconds <= _EPOCH_MAX):
        return None
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)


def _find_file(directory: Path, name: str) -> Path | None:
    """Return *name* in *directory*, matching case-insensitively."""
    exact = directory / name
    if exact.is_file():
        return exact
    try:
        for p in directory.iterdir():
            if p.name.lower() == name and p.is_file():
                return p
    except OSError:
        return None
    return None


def _companion_paths(path: Path) -> tuple[Path, Path] | None:
    """Resolve (patient.hea, ecgraw.dat) from a directory or either file."""
    if path.is_dir():
        hea, dat = _find_file(path, _HEA_NAME), _find_file(path, _DAT_NAME)
    else:
        name = path.name.lower()
        if name == _HEA_NAME:
            hea, dat = path, _find_file(path.parent, _DAT_NAME)
        elif name == _DAT_NAME:
            hea, dat = _find_file(path.parent, _HEA_NAME), path
        else:
            return None
    if hea is None or dat is None:
        return None
    return hea, dat


def _looks_like_patient_hea(buf: bytes, off: int) -> bool:
    """Structural fingerprint for a patient.hea header at offset *off*.

    Combines five checks:

    * channel count (byte 12) is 3 or 12,
    * sampling rate (s16 at 32) is one of the SE-2012 rates,
    * both timestamps (u32 at 4 and 8) decode to plausible epochs,
    * the recording length implied by the timestamps is below 14 days,
    * the first lead label slot at offset 1796 contains ASCII bytes.

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
    label_off = off + _LABEL_OFFSET
    if label_off + _LABEL_SIZE <= len(buf):
        label = buf[label_off:label_off + _LABEL_SIZE].split(b"\x00", 1)[0]
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

    The literal must sit at its documented offset (0x04); accepting any
    location risks colliding with a coincidental in-payload occurrence.
    """
    end = _NEUTRAL_HOLTER_SIGNATURE_OFFSET + len(_NEUTRAL_HOLTER_SIGNATURE)
    if len(arc) < end:
        return False
    return (
        arc[_NEUTRAL_HOLTER_SIGNATURE_OFFSET:end]
        == _NEUTRAL_HOLTER_SIGNATURE
    )


def _arc_has_signature(header: bytes) -> bool:
    """True when the first bytes of an .arc file look like an EDAN archive."""
    if _is_neutral_holter(header):
        return True
    if _HEA_NAME.encode("ascii") in header:
        return True
    return any(
        _looks_like_patient_hea(header, off)
        for off in range(0, max(0, len(header) - 64), 4)
    )


def _find_neutral_holter_payload_end(
    arc: bytes,
    start: int,
    channels: int,
) -> int:
    """Locate the boundary between the ECG samples and the beat-records.

    ECG samples have very low int16 std (~25); the trailing 32-byte beat
    records yield std > 1000. Scan forward in blocks looking for the
    first block whose std crosses the threshold, then refine to a
    channel-aligned offset.
    """
    cursor = start
    block = _NEUTRAL_SCAN_BLOCK
    # Default to "no jump found": take the rest of the file as signal.
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
    Returns ``None`` if the pattern doesn't match.
    """
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


def _payload_start(arc: bytes, start: int) -> int | None:
    """First offset in ``[start, start + 256)`` where ADC samples begin.

    EDAN samples are uint16 LE centred at 16384 with a small range, so
    16 consecutive values inside (10000, 24000) mark the payload start.
    Wrapper bytes (NULs, size fields) decode far from mid-scale.
    """
    for off in range(start, start + 256):
        if off + 32 > len(arc):
            return None
        probe = np.frombuffer(arc[off:off + 32], dtype="<u2")
        if np.all((probe > 10_000) & (probe < 24_000)):
            return off
    return None


def _scan_for_ecgraw_dat(arc: bytes, hea_end: int) -> bytes:
    """Locate the embedded ecgraw.dat payload in an .arc archive.

    Strategy:
    1. Look for the ``ecgraw.dat`` ASCII filename marker and take the
       bytes from the first ADC-like sample after it.
    2. Otherwise take the bytes after the patient.hea header, starting
       at the first ADC-like sample.
    """
    pos = arc.find(_DAT_NAME.encode("ascii"))
    if pos != -1:
        start = _payload_start(arc, pos + len(_DAT_NAME))
        if start is not None:
            return arc[start:]
    start = _payload_start(arc, hea_end)
    if start is None:
        start = hea_end
    payload = arc[start:]
    if len(payload) < 2:
        raise CorruptedFileError(
            "No usable ecgraw.dat payload found in .arc after patient.hea"
        )
    return payload


def _plausible_columns(matrix: np.ndarray) -> bool:
    """True when every column's median sits near ADC zero."""
    medians = np.median(matrix, axis=0)
    return bool(np.all(np.abs(medians - _ADC_ZERO) <= _ADC_PLAUSIBLE_RANGE))


def _roughness(matrix: np.ndarray) -> float:
    """Mean absolute sample-to-sample step over the columns of *matrix*."""
    if matrix.shape[0] < 2:
        return float("inf")
    return float(np.mean(np.abs(np.diff(matrix, axis=0))))


def _choose_stride(
    raw: np.ndarray,
    channel_count: int,
    sampling_rate: int,
    header_seconds: float | None,
) -> tuple[int, str]:
    """Infer how many channels are interleaved in ecgraw.dat.

    The header stores *channel_count*, but the reference says all 12
    channels are always written. Candidates are therefore the header
    count and 12, judged in order by:

    1. ADC plausibility: the first *channel_count* columns of a correct
       layout are centred on ADC zero; a wrong stride mixes in padding
       channels and breaks this.
    2. The header duration (start/end epochs) against the frame count.
    3. Constant padding in columns ``channel_count..11`` of a 12-wide
       layout.
    4. Signal smoothness: a wrong stride mixes channels or skips frames,
       which increases sample-to-sample steps.

    Returns ``(stride, reason)``. Raises :class:`CorruptedFileError` when
    no candidate is consistent with the data.
    """
    candidates = sorted({channel_count, _MAX_CHANNELS})
    probes: dict[int, np.ndarray] = {}
    for s in candidates:
        frames = raw.size // s
        if frames == 0:
            continue
        n = min(frames, _STRIDE_PROBE_FRAMES)
        matrix = raw[: n * s].reshape(n, s).astype(np.float64)
        if _plausible_columns(matrix[:, :channel_count]):
            probes[s] = matrix
    if not probes:
        raise CorruptedFileError(
            f"ecgraw.dat ({raw.size} samples) is not consistent with "
            f"{' or '.join(str(s) for s in candidates)} interleaved channels "
            f"centred on ADC zero {_ADC_ZERO}"
        )
    if len(probes) == 1:
        return next(iter(probes)), "single plausible layout"

    if header_seconds is not None:
        expected = header_seconds * sampling_rate
        tol = max(2 * sampling_rate, 0.01 * expected)
        matching = [s for s in probes if abs(raw.size // s - expected) <= tol]
        if len(matching) == 1:
            return matching[0], "header duration"

    wide = probes.get(_MAX_CHANNELS)
    if wide is not None:
        padding = wide[:, channel_count:]
        active = wide[:, :channel_count]
        if np.all(np.ptp(padding, axis=0) == 0) and np.any(np.ptp(active, axis=0) > 0):
            return _MAX_CHANNELS, "constant padding channels"

    rough = {s: _roughness(m[:, :channel_count]) for s, m in probes.items()}
    best = min(rough, key=rough.get)
    other = max(rough, key=rough.get)
    if rough[best] < 0.5 * rough[other]:
        return best, "signal smoothness"
    return channel_count, "ambiguous (header channel count used)"


class EDANARCHolterParser(Parser):
    """Parser for EDAN SE-2012 / ARC Holter recordings.

    Accepts these entry points:

    * a recording directory, ``patient.hea`` or ``ecgraw.dat`` (the
      companion file must sit in the same directory);
    * ``*.arc`` archives, best-effort heuristic extraction (emits a
      warning).
    """

    FORMAT_NAME = "EDAN ARC Holter"
    FORMAT_DESCRIPTION = "EDAN SE-2012 / ARC Holter (patient.hea + ecgraw.dat or .arc)"
    FILE_EXTENSIONS = [".hea", ".dat", ".arc"]
    PRIORITY = 40

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        if file_path.name.lower().endswith(_ARC_EXT) and file_path.is_file():
            return _arc_has_signature(header)
        return _companion_paths(file_path) is not None

    def parse(self, file_path: Path) -> ECGRecord:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        notes: list[str] = []
        if path.is_file() and path.name.lower().endswith(_ARC_EXT):
            arc = path.read_bytes()
            if _is_neutral_holter(arc):
                warnings.warn(
                    "NEUTRAL HOLTER RECORDING .arc parsing is reverse-engineered "
                    "from a single sample file; no vendor specification exists. "
                    "The signal layout (3 channels x int16 LE @ 250 Hz interleaved) "
                    "and metadata are best guesses. Validate sample values and "
                    "channel order against the recorder's own viewer before any "
                    "clinical or research use.",
                    UserWarning,
                    stacklevel=2,
                )
                record = self._parse_neutral_holter(path, arc)
                fill_signal_summary(record)
                return record
            warnings.warn(
                "Parsing EDAN .arc archives is best-effort: the wrapper format "
                "is undocumented. Sample values and metadata should be sanity-"
                "checked against the recorder's own export.",
                UserWarning,
                stacklevel=2,
            )
            hea_bytes, dat_bytes = self._read_arc(arc)
            record = self._build_record(hea_bytes, dat_bytes, "edan_arc_archive", notes)
            record.raw_metadata["arc_filepath"] = str(path)
            record.raw_metadata["arc_heuristic"] = True
        else:
            pair = _companion_paths(path)
            if pair is None:
                raise CorruptedFileError(
                    f"EDAN recording needs both {_HEA_NAME} and {_DAT_NAME} "
                    f"in the same directory: {path}"
                )
            hea_path, dat_path = pair
            record = self._build_record(
                hea_path.read_bytes(), dat_path.read_bytes(), "edan_arc", notes,
            )
            record.raw_metadata["hea_filepath"] = str(hea_path)
            record.raw_metadata["data_filepath"] = str(dat_path)

        record.raw_metadata["filepath"] = str(path)
        for note in notes:
            warnings.warn(note, UserWarning, stacklevel=2)
        fill_signal_summary(record)
        return record

    # I/O helpers

    @staticmethod
    def _read_arc(arc: bytes) -> tuple[bytes, bytes]:
        if len(arc) < 64:
            raise CorruptedFileError(
                f"EDAN .arc too small: {len(arc)} bytes"
            )
        hea_off = _scan_for_patient_hea(arc)
        hea_end = hea_off + _HEA_SIZE
        hea_bytes = arc[hea_off:hea_end]
        dat_bytes = _scan_for_ecgraw_dat(arc, hea_end)
        return hea_bytes, dat_bytes

    # NEUTRAL HOLTER decode (reverse-engineered, single sample)

    def _parse_neutral_holter(self, arc_path: Path, arc: bytes) -> ECGRecord:
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
        # "DT-06_05_2026-11_38_39.arc" -> 2026-05-06 11:38:39.
        recording_start = _parse_filename_timestamp(arc_path.name)
        if recording_start is not None:
            record.recording.date = recording_start

        record.raw_metadata["filepath"] = str(arc_path)
        record.raw_metadata["arc_filepath"] = str(arc_path)
        record.raw_metadata["arc_variant"] = "neutral_holter"
        record.raw_metadata["reverse_engineered"] = True
        # The 250 Hz rate and 3 channels are not read from the file
        record.raw_metadata["sampling_rate_stated"] = False
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
    def _build_record(
        hea: bytes, dat: bytes, source_format: str, notes: list[str],
    ) -> ECGRecord:
        """Decode a patient.hea / ecgraw.dat pair; warnings go to *notes*."""
        if len(hea) < 64:
            raise CorruptedFileError(
                f"EDAN patient.hea too small: {len(hea)} bytes"
            )

        channel_count = hea[12]
        if not (1 <= channel_count <= _MAX_CHANNELS):
            raise CorruptedFileError(
                f"Invalid EDAN channel count: {channel_count}"
            )
        labels_end = _LABEL_OFFSET + _LABEL_SIZE * channel_count
        if len(hea) < labels_end:
            raise CorruptedFileError(
                f"EDAN patient.hea truncated: {len(hea)} bytes, the "
                f"{channel_count} channel labels end at byte {labels_end}"
            )
        if len(hea) < _HEA_SIZE:
            notes.append(
                f"EDAN patient.hea is truncated ({len(hea)} of {_HEA_SIZE} "
                "bytes); fields past the end are left empty."
            )

        sampling_rate = _s16(hea, 32)
        if not (0 < sampling_rate <= _MAX_SAMPLING_RATE):
            raise CorruptedFileError(
                f"Invalid EDAN sampling rate: {sampling_rate}"
            )
        if sampling_rate not in _VALID_SAMPLING_RATES:
            notes.append(f"Unusual EDAN sampling rate: {sampling_rate} Hz")

        # Timestamps (Unix epoch seconds at offsets 4 and 8)
        start_epoch = _u32(hea, 4)
        end_epoch = _u32(hea, 8)
        start_dt = _epoch_to_datetime(start_epoch)
        end_dt = _epoch_to_datetime(end_epoch)
        if start_dt is None:
            notes.append(
                f"EDAN start timestamp {start_epoch} is not a plausible date; "
                "recording date left empty."
            )
        if end_dt is not None and (start_dt is None or end_dt <= start_dt):
            notes.append(
                f"EDAN end timestamp {end_epoch} is not after the start "
                f"timestamp {start_epoch}; ignored."
            )
            end_dt = None
        elif end_dt is None and end_epoch:
            notes.append(
                f"EDAN end timestamp {end_epoch} is not a plausible date; ignored."
            )
        header_seconds: float | None = None
        if start_dt and end_dt:
            header_seconds = (end_dt - start_dt).total_seconds()
            if header_seconds > _MAX_HEADER_SECONDS:
                header_seconds = None

        # Signal data
        if len(dat) % 2:
            notes.append("EDAN ecgraw.dat has an odd byte count; last byte ignored.")
            dat = dat[:-1]
        raw = np.frombuffer(dat, dtype="<u2")
        if raw.size < channel_count:
            raise CorruptedFileError(
                f"EDAN signal payload contains no samples for "
                f"{channel_count} channels"
            )
        stride, stride_reason = _choose_stride(
            raw, channel_count, sampling_rate, header_seconds,
        )
        if stride_reason.startswith("ambiguous"):
            notes.append(
                f"EDAN channel layout is ambiguous ({channel_count} or "
                f"{_MAX_CHANNELS} interleaved channels); using the header "
                f"count {channel_count}."
            )
        samples_per_channel = raw.size // stride
        leftover = raw.size - samples_per_channel * stride
        if leftover:
            notes.append(
                f"EDAN ecgraw.dat ends with a partial frame ({leftover} of "
                f"{stride} samples); ignored."
            )
        matrix = raw[: samples_per_channel * stride].reshape(samples_per_channel, stride)
        signed = matrix[:, :channel_count].astype(np.float64) - _ADC_ZERO

        data_seconds = samples_per_channel / sampling_rate
        duration = timedelta(seconds=data_seconds)
        if header_seconds is not None:
            tol = max(2.0, 0.01 * header_seconds)
            if abs(data_seconds - header_seconds) > tol:
                notes.append(
                    f"EDAN header timestamps span {header_seconds:.0f} s but "
                    f"ecgraw.dat holds {data_seconds:.1f} s of signal; the "
                    "duration is taken from the signal."
                )

        # Patient demographics
        patient = PatientInfo()
        patient.patient_id = _text(hea, 108, 32)
        diagnosis = _text(hea, 140, 102)
        medication = _text(hea, 242, 102)
        medical_history = _text(hea, 2892, 64)
        # The name field is followed by the physician field at 2700.
        full_name = _text(hea, 2637, 2700 - 2637)
        if full_name:
            parts = full_name.split(None, 1)
            if len(parts) == 2:
                patient.first_name, patient.last_name = parts
            else:
                patient.last_name = full_name
        height = _s16(hea, 60)
        weight = _s16(hea, 64)
        if height > 0:
            patient.height = float(height)
        if weight > 0:
            patient.weight = float(weight)
        if medication:
            patient.medications = [medication]
        patient.clinical_history = "\n".join(
            t for t in (diagnosis, medical_history) if t
        )

        # Recording / device metadata
        recording = RecordingInfo()
        recording.date = start_dt
        recording.duration = duration
        recording.end_date = end_dt
        recording.referring_physician = _text(hea, 2700, 64)
        recording.technician = _text(hea, 2764, 64)

        recorder_id = _text(hea, 2304, 10)
        version_a = _text(hea, 2314, 6)
        version_b = _text(hea, 2416, 6)
        recording.device = DeviceInfo(
            manufacturer="EDAN",
            serial_number=recorder_id,
            software_version=version_a or version_b,
            department=_text(hea, 1960, 134),
            acquisition_type="Holter",
        )

        # Per-channel electrode labels (8 bytes each from offset 1796)
        header_labels = [
            _text(hea, _LABEL_OFFSET + ch * _LABEL_SIZE, _LABEL_SIZE)
            for ch in range(channel_count)
        ]
        labels = unique_labels([
            normalize_lead_label(lbl) if lbl else f"Ch{ch + 1}"
            for ch, lbl in enumerate(header_labels)
        ])

        record = ECGRecord(source_format=source_format)
        record.patient = patient
        record.recording = recording
        for ch in range(channel_count):
            record.leads.append(Lead(
                label=labels[ch],
                samples=np.ascontiguousarray(signed[:, ch]),
                sampling_rate=sampling_rate,
            ))

        record.recording.acquisition.signal = SignalCharacteristics(
            sampling_rate=sampling_rate,
            bits_per_sample=16,
            signal_offset=_ADC_ZERO,
            signal_signed=False,
            number_channels_allocated=stride,
            number_channels_valid=channel_count,
            data_encoding="uint16",
            compression="none",
        )
        lowpass = _s16(hea, 2596)
        if lowpass > 0:
            record.recording.acquisition.filters = FilterSettings(lowpass=float(lowpass))

        meta = record.raw_metadata
        meta["start_epoch"] = int(start_epoch)
        meta["end_epoch"] = int(end_epoch)
        meta["epoch_timezone"] = "unverified (decoded as UTC)"
        meta["header_size"] = len(hea)
        meta["header_channel_count"] = channel_count
        meta["stored_channel_count"] = stride
        meta["channel_layout_basis"] = stride_reason
        meta["samples_per_channel"] = samples_per_channel
        meta["patient_name"] = full_name
        meta["diagnosis"] = diagnosis
        meta["medical_history"] = medical_history
        meta["telephone"] = _text(hea, 68, 40)
        meta["accession_number"] = _text(hea, 344, 68)
        meta["in_out_pe_id"] = _text(hea, 412, 134)
        meta["patient_area"] = _text(hea, 546, 134)
        meta["recorder_id"] = recorder_id
        meta["version_2314"] = version_a
        meta["version_2416"] = version_b
        meta["dft_filter"] = _text(hea, 2628, 5)
        meta["procedure"] = _text(hea, 2828, 64)
        meta["address"] = _text(hea, 2956, 86)
        meta["lead_labels"] = header_labels
        return record

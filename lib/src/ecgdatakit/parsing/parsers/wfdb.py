"""WFDB (PhysioNet/MIT format) parser.

Parses ``.hea`` header files and their signal files.

Reference: https://physionet.org/physiotools/wag/header-5.htm
           https://physionet.org/physiotools/wag/signal-5.htm

Supported storage formats: 0 (null), 8, 16, 24, 32, 61, 80, 160, 212, 310
and 311, with samples per frame, skew and byte offset. Multi-segment
records (fixed and variable layout) are concatenated. FLAC formats (508,
516, 524) are not supported.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np

from ecgdatakit.exceptions import (
    ChecksumWarning,
    CorruptedFileError,
    UnsupportedFormatError,
)
from ecgdatakit.models import (
    ECGRecord,
    FileFormatInfo,
    Interpretation,
    Lead,
    PatientInfo,
    RecordingInfo,
    SignalCharacteristics,
    _normalize_unit,
    derive_is_raw,
)
from ecgdatakit.parsing.helpers import (
    decode_text,
    fill_signal_summary,
    normalize_lead_label,
    unique_labels,
)
from ecgdatakit.parsing.parser import Parser

_RECORD_LINE_RE = re.compile(r"^[\w\-]+(?:/\d+)?\s+\d+(?:\s|$)")
_FORMAT_RE = re.compile(r"^(\d+)(?:x(\d+))?(?::(\d+))?(?:\+(\d+))?$")
_GAIN_RE = re.compile(
    r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)(?:\((-?\d+)\))?(?:/(\S+))?$"
)
_MIT_AGE_SEX_RE = re.compile(r"^(\d{1,3}|\?)\s+([MF?])(?:\s|$)")

# Bits per sample of each storage format (used when ADC resolution is 0)
_FORMAT_BITS = {
    0: 0, 8: 8, 16: 16, 24: 24, 32: 32, 61: 16, 80: 8, 160: 16,
    212: 12, 310: 10, 311: 10,
}
# Sample value WFDB reserves for "invalid sample" in each format
_INVALID_SAMPLE = {
    16: -32768, 24: -(2 ** 23), 32: -(2 ** 31), 61: -32768,
    80: -128, 160: -32768, 212: -2048, 310: -512, 311: -512,
}
_FLAC_FORMATS = {508, 516, 524}


@dataclass
class _SignalSpec:
    filename: str
    fmt: int
    samples_per_frame: int = 1
    skew: int = 0
    byte_offset: int = 0
    gain: float = 0.0  # ADC units per physical unit, 0 = uncalibrated
    baseline: int = 0
    units: str = "mV"
    adc_resolution: int = 0
    adc_zero: int = 0
    initial_value: int = 0
    checksum: int | None = None
    block_size: int = 0
    description: str = ""


@dataclass
class _Header:
    record_name: str
    num_signals: int
    sampling_rate: float = 250.0
    sampling_rate_stated: bool = False
    counter_frequency: float | None = None
    base_counter: float | None = None
    num_samples: int = 0
    base_time: time | None = None
    base_date: datetime | None = None
    num_segments: int | None = None
    segments: list[tuple[str, int]] = field(default_factory=list)
    signals: list[_SignalSpec] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)


def _fail(msg: str) -> CorruptedFileError:
    return CorruptedFileError(f"WFDB header: {msg}")


def _parse_int(text: str, what: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise _fail(f"invalid {what} {text!r}") from None


def _parse_float(text: str, what: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise _fail(f"invalid {what} {text!r}") from None


def _parse_base_time(text: str) -> time:
    """Parse ``HH:MM:SS[.ffffff]`` (``MM:SS`` and ``SS`` are also accepted)."""
    parts = text.split(":")
    if not 1 <= len(parts) <= 3:
        raise _fail(f"invalid base time {text!r}")
    parts = ["0"] * (3 - len(parts)) + parts
    try:
        hour, minute = int(parts[0]), int(parts[1])
        seconds = float(parts[2])
        sec = int(seconds)
        micro = round((seconds - sec) * 1_000_000)
        return time(hour, minute, sec, min(micro, 999_999))
    except ValueError:
        raise _fail(f"invalid base time {text!r}") from None


def _parse_record_line(line: str) -> _Header:
    tokens = line.split()
    if len(tokens) < 2:
        raise _fail(f"record line too short: {line!r}")
    name, _, nseg = tokens[0].partition("/")
    header = _Header(record_name=name, num_signals=_parse_int(tokens[1], "signal count"))
    if header.num_signals < 0:
        raise _fail(f"negative signal count {header.num_signals}")
    if nseg:
        header.num_segments = _parse_int(nseg, "segment count")
        if header.num_segments <= 0:
            raise _fail(f"invalid segment count {header.num_segments}")
    rest = tokens[2:]
    if rest:
        m = re.match(r"^([\d.eE+\-]+)(?:/([\d.eE+\-]+)(?:\(([\d.eE+\-]+)\))?)?$", rest[0])
        if not m:
            raise _fail(f"invalid sampling frequency {rest[0]!r}")
        header.sampling_rate = _parse_float(m.group(1), "sampling frequency")
        if m.group(2):
            header.counter_frequency = _parse_float(m.group(2), "counter frequency")
        if m.group(3):
            header.base_counter = _parse_float(m.group(3), "base counter")
        if header.sampling_rate <= 0:
            header.sampling_rate = 250.0
        else:
            header.sampling_rate_stated = True
    if len(rest) > 1:
        header.num_samples = _parse_int(rest[1], "number of samples")
        if header.num_samples < 0:
            raise _fail(f"negative number of samples {header.num_samples}")
    if len(rest) > 2:
        header.base_time = _parse_base_time(rest[2])
    if len(rest) > 3:
        try:
            header.base_date = datetime.strptime(rest[3], "%d/%m/%Y")
        except ValueError:
            raise _fail(f"invalid base date {rest[3]!r}") from None
    return header


def _parse_signal_line(line: str) -> _SignalSpec:
    """Parse ``file format[xspf][:skew][+offset] gain[(baseline)][/units]
    adc_res adc_zero init_value checksum block_size description``."""
    tokens = line.split()
    if len(tokens) < 2:
        raise _fail(f"signal line too short: {line!r}")
    m = _FORMAT_RE.match(tokens[1])
    if not m:
        raise _fail(f"invalid format field {tokens[1]!r}")
    spec = _SignalSpec(filename=tokens[0], fmt=int(m.group(1)))
    if m.group(2) is not None:
        spec.samples_per_frame = int(m.group(2))
        if spec.samples_per_frame <= 0:
            raise _fail(f"invalid samples per frame in {tokens[1]!r}")
    if m.group(3) is not None:
        spec.skew = int(m.group(3))
    if m.group(4) is not None:
        spec.byte_offset = int(m.group(4))

    baseline: int | None = None
    if len(tokens) > 2:
        g = _GAIN_RE.match(tokens[2])
        if not g:
            raise _fail(f"invalid ADC gain field {tokens[2]!r}")
        spec.gain = float(g.group(1))
        if g.group(2) is not None:
            baseline = int(g.group(2))
        if g.group(3):
            spec.units = g.group(3)
    if len(tokens) > 3:
        spec.adc_resolution = _parse_int(tokens[3], "ADC resolution")
    if len(tokens) > 4:
        spec.adc_zero = _parse_int(tokens[4], "ADC zero")
    # Baseline and initial value default to the ADC zero
    spec.baseline = spec.adc_zero if baseline is None else baseline
    spec.initial_value = spec.adc_zero
    if len(tokens) > 5:
        spec.initial_value = _parse_int(tokens[5], "initial value")
    if len(tokens) > 6:
        spec.checksum = _parse_int(tokens[6], "checksum")
    if len(tokens) > 7:
        spec.block_size = _parse_int(tokens[7], "block size")
    if len(tokens) > 8:
        spec.description = " ".join(tokens[8:])
    if not spec.adc_resolution:
        spec.adc_resolution = _FORMAT_BITS.get(spec.fmt, 0)
    return spec


def _parse_header_text(text: str) -> _Header:
    lines: list[str] = []
    comments: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments.append(line[1:].strip())
        else:
            lines.append(line)
    if not lines:
        raise CorruptedFileError("WFDB header: no record line")
    header = _parse_record_line(lines[0])
    header.comments = comments
    body = lines[1:]
    if header.num_segments is not None:
        if len(body) < header.num_segments:
            raise _fail(
                f"{header.num_segments} segments declared, {len(body)} segment lines found"
            )
        for line in body[:header.num_segments]:
            tokens = line.split()
            if len(tokens) < 2:
                raise _fail(f"invalid segment line {line!r}")
            header.segments.append((tokens[0], _parse_int(tokens[1], "segment length")))
        return header
    if len(body) < header.num_signals:
        raise _fail(
            f"{header.num_signals} signals declared, {len(body)} signal lines found"
        )
    header.signals = [_parse_signal_line(line) for line in body[:header.num_signals]]
    return header


def _decode_samples(data: bytes, fmt: int) -> np.ndarray:
    """Decode a byte stream into the sequence of stored sample values."""
    buf = np.frombuffer(data, dtype=np.uint8)
    if fmt in (8, 80):
        if fmt == 8:
            return buf.view(np.int8).astype(np.int64)
        return buf.astype(np.int64) - 128
    if fmt in (16, 61, 160):
        n = len(buf) // 2
        dtype = ">i2" if fmt == 61 else ("<u2" if fmt == 160 else "<i2")
        values = np.frombuffer(data[:2 * n], dtype=dtype).astype(np.int64)
        return values - 32768 if fmt == 160 else values
    if fmt == 24:
        n = len(buf) // 3
        b = buf[:3 * n].reshape(n, 3).astype(np.int64)
        values = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        return np.where(values >= 1 << 23, values - (1 << 24), values)
    if fmt == 32:
        n = len(buf) // 4
        return np.frombuffer(data[:4 * n], dtype="<i4").astype(np.int64)
    if fmt == 212:
        # Two 12-bit samples in three bytes; a trailing byte pair holds one
        n = len(buf) // 3
        b = buf[:3 * n].reshape(n, 3).astype(np.int64)
        out = np.empty(2 * n, dtype=np.int64)
        out[0::2] = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
        out[1::2] = b[:, 2] | ((b[:, 1] & 0xF0) << 4)
        if len(buf) - 3 * n >= 2:
            tail = buf[3 * n:].astype(np.int64)
            out = np.append(out, tail[0] | ((tail[1] & 0x0F) << 8))
        return np.where(out >= 2048, out - 4096, out)
    if fmt in (310, 311):
        # Three 10-bit samples in four bytes
        n = len(buf) // 4
        b = buf[:4 * n].reshape(n, 4).astype(np.int64)
        out = np.empty(3 * n, dtype=np.int64)
        if fmt == 310:
            w0 = b[:, 0] | (b[:, 1] << 8)
            w1 = b[:, 2] | (b[:, 3] << 8)
            out[0::3] = (w0 >> 1) & 0x3FF
            out[1::3] = (w1 >> 1) & 0x3FF
            out[2::3] = ((w0 >> 11) & 0x1F) | (((w1 >> 11) & 0x1F) << 5)
        else:
            w = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24)
            out[0::3] = w & 0x3FF
            out[1::3] = (w >> 10) & 0x3FF
            out[2::3] = (w >> 20) & 0x3FF
        return np.where(out >= 512, out - 1024, out)
    if fmt in _FLAC_FORMATS:
        raise UnsupportedFormatError(f"WFDB format {fmt} (FLAC) is not supported")
    raise UnsupportedFormatError(f"Unknown WFDB storage format {fmt}")


def _checksum_ok(values: np.ndarray, stored: int) -> bool:
    """Compare the 16-bit sum of the samples with the stored checksum.

    Some writers store the checksum unsigned, so compare modulo 2**16.
    """
    total = int(np.sum(values, dtype=np.int64))
    return (total - stored) & 0xFFFF == 0


class WFDBParser(Parser):
    """Parser for WFDB (PhysioNet/MIT) ECG files."""

    FORMAT_NAME = "WFDB"
    FORMAT_DESCRIPTION = "WaveForm DataBase / PhysioNet MIT format"
    FILE_EXTENSIONS = [".hea", ".dat"]
    PRIORITY = 40

    @staticmethod
    def _first_record_line(header: bytes) -> str | None:
        text = header.decode("latin-1")
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
        return None

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        suffix = file_path.suffix.lower()
        if suffix == ".dat":
            # Signal file: accept it when a matching header sits next to it
            hea_path = file_path.with_suffix(".hea")
            if not hea_path.is_file():
                return False
            with open(hea_path, "rb") as f:
                header = f.read(4096)
        elif suffix != ".hea":
            return False
        line = WFDBParser._first_record_line(header)
        return bool(line and _RECORD_LINE_RE.match(line))

    def parse(self, file_path: Path) -> ECGRecord:
        path = Path(file_path)
        hea_path = path if path.suffix.lower() == ".hea" else path.with_suffix(".hea")
        if not hea_path.exists():
            raise FileNotFoundError(f"Header file not found: {hea_path}")

        self._path = path
        self._warnings: list[tuple[str, type[Warning]]] = []
        self._checksums: list[bool] = []
        self._bad_checksums: list[str] = []
        self._formats: set[int] = set()
        header = _parse_header_text(decode_text(hea_path.read_bytes()))

        record = ECGRecord(source_format="wfdb")
        if header.num_segments is not None:
            specs, signals, sig_len = self._read_multi_segment(hea_path, header)
        else:
            specs = header.signals
            signals = self._read_signals(hea_path.parent, header)
            sig_len = header.num_samples or max(
                (len(s) // sp.samples_per_frame for s, sp in zip(signals, specs)), default=0
            )

        record.leads = self._build_leads(header, specs, signals)
        self._fill_metadata(record, header, specs, sig_len)

        for msg, category in self._warnings:
            warnings.warn(msg, category, stacklevel=2)
        fill_signal_summary(record)
        return record

    # ------------------------------------------------------------------
    # Signal reading
    # ------------------------------------------------------------------

    def _read_signals(self, directory: Path, header: _Header) -> list[np.ndarray]:
        """Return one float64 array of stored values per signal, header order."""
        specs = header.signals
        n = header.num_samples
        result: list[np.ndarray | None] = [None] * len(specs)

        # Signals sharing a file are interleaved frame by frame in header order
        files: dict[str, list[int]] = {}
        for i, spec in enumerate(specs):
            if spec.fmt == 0:
                # Null signal: not stored in any file
                result[i] = np.full(n, np.nan)
                continue
            files.setdefault(spec.filename, []).append(i)

        for filename, indices in files.items():
            group = [specs[i] for i in indices]
            fmts = {s.fmt for s in group}
            if len(fmts) > 1:
                raise _fail(f"signals in {filename} use several formats {sorted(fmts)}")
            fmt = group[0].fmt
            self._formats.add(fmt)
            if fmt in _FLAC_FORMATS:
                raise UnsupportedFormatError(f"WFDB format {fmt} (FLAC) is not supported")
            if fmt not in _FORMAT_BITS:
                raise UnsupportedFormatError(f"Unknown WFDB storage format {fmt}")
            dat_path = directory / filename
            if filename == "~" or not dat_path.is_file():
                raise CorruptedFileError(f"WFDB signal file not found: {dat_path}")
            data = dat_path.read_bytes()
            byte_offset = group[0].byte_offset
            if byte_offset > len(data):
                raise CorruptedFileError(
                    f"WFDB byte offset {byte_offset} exceeds size of {filename}"
                )
            stream = _decode_samples(data[byte_offset:], fmt)

            spf = [s.samples_per_frame for s in group]
            frame_size = sum(spf)
            n_frames = len(stream) // frame_size
            max_skew = max(s.skew for s in group)
            if n and n_frames < n:
                raise CorruptedFileError(
                    f"WFDB signal file {filename} holds {n_frames} frames, "
                    f"header declares {n}"
                )
            length = n if n else max(n_frames - max_skew, 0)
            frames = stream[:n_frames * frame_size].reshape(n_frames, frame_size)

            col = 0
            for i, spec in zip(indices, group):
                values = frames[:, col:col + spec.samples_per_frame].astype(np.float64)
                col += spec.samples_per_frame
                if fmt == 8:
                    values = np.cumsum(values.ravel()).reshape(values.shape) + spec.initial_value
                # Skewed samples: sample 0 of the signal is in frame `skew`
                values = values[spec.skew:spec.skew + length]
                if len(values) < length:
                    pad = np.full((length - len(values), spec.samples_per_frame), np.nan)
                    values = np.vstack([values, pad])
                result[i] = values.ravel()
                if spec.checksum is not None and n and not np.isnan(values).any():
                    ok = _checksum_ok(values.astype(np.int64), spec.checksum)
                    self._checksums.append(ok)
                    if not ok:
                        self._bad_checksums.append(spec.description or f"{filename}#{i}")
        return [r if r is not None else np.empty(0) for r in result]

    def _read_multi_segment(
        self, hea_path: Path, header: _Header
    ) -> tuple[list[_SignalSpec], list[np.ndarray], int]:
        """Concatenate the segments of a multi-segment record.

        Fixed-layout records keep the first segment's signal arrangement.
        Variable-layout records take it from the layout segment (segment 0
        with 0 samples); signals absent from a segment are NaN and signals
        whose gain or baseline differ are rescaled to the layout values.
        """
        directory = hea_path.parent
        segments = list(header.segments)
        layout: list[_SignalSpec] | None = None
        variable = bool(segments) and segments[0][1] == 0
        if variable:
            layout = self._read_segment_header(directory, segments[0][0]).signals
            segments = segments[1:]

        parts: list[tuple[int, list[_SignalSpec] | None, list[np.ndarray] | None]] = []
        for name, length in segments:
            if name == "~":
                parts.append((length, None, None))
                continue
            seg = self._read_segment_header(directory, name)
            if seg.num_samples and seg.num_samples != length:
                raise _fail(
                    f"segment {name} has {seg.num_samples} samples, "
                    f"master header declares {length}"
                )
            seg.num_samples = length
            parts.append((length, seg.signals, self._read_signals(directory, seg)))
            if layout is None:
                layout = seg.signals
        if layout is None:
            raise _fail("multi-segment record has only null segments")

        out: list[list[np.ndarray]] = [[] for _ in layout]
        for length, seg_specs, seg_signals in parts:
            for k, target in enumerate(layout):
                n = length * target.samples_per_frame
                src = None
                if seg_specs is not None:
                    if variable:
                        src = next(
                            (j for j, s in enumerate(seg_specs)
                             if s.description == target.description), None)
                    elif k < len(seg_specs):
                        src = k
                if src is None:
                    out[k].append(np.full(n, np.nan))
                    continue
                spec, values = seg_specs[src], seg_signals[src]
                if spec.samples_per_frame != target.samples_per_frame:
                    raise UnsupportedFormatError(
                        "WFDB multi-segment record changes samples per frame "
                        f"of signal {target.description or k}"
                    )
                if (spec.gain, spec.baseline) != (target.gain, target.baseline) and spec.gain:
                    values = (values - spec.baseline) / spec.gain * (
                        target.gain or spec.gain) + target.baseline
                out[k].append(values)
        signals = [np.concatenate(chunks) if chunks else np.empty(0) for chunks in out]
        return layout, signals, sum(length for length, _, _ in parts)

    def _read_segment_header(self, directory: Path, name: str) -> _Header:
        seg_path = directory / f"{name}.hea"
        if not seg_path.is_file():
            raise CorruptedFileError(f"WFDB segment header not found: {seg_path}")
        seg = _parse_header_text(decode_text(seg_path.read_bytes()))
        if seg.num_segments is not None:
            raise UnsupportedFormatError("Nested WFDB multi-segment records are not supported")
        return seg

    # ------------------------------------------------------------------
    # Record assembly
    # ------------------------------------------------------------------

    def _build_leads(
        self, header: _Header, specs: list[_SignalSpec], signals: list[np.ndarray]
    ) -> list[Lead]:
        labels = unique_labels([
            normalize_lead_label(s.description) if s.description else f"Ch{i}"
            for i, s in enumerate(specs)
        ])
        leads: list[Lead] = []
        invalid: dict[str, int] = {}
        for label, spec, values in zip(labels, specs, signals):
            bad = _INVALID_SAMPLE.get(spec.fmt)
            if bad is not None:
                count = int(np.count_nonzero(values == bad))
                if count:
                    invalid[label] = count

            if spec.gain:
                unit = _normalize_unit(spec.units) or spec.units
                resolution = 1.0 / spec.gain
                offset = -spec.baseline / spec.gain
                adc_unit = f"adu/{spec.units}"
            else:
                # Gain 0 or missing: the signal is uncalibrated
                unit, resolution, offset, adc_unit = "", 1.0, 0.0, ""
            is_raw = derive_is_raw(resolution, offset, unit)
            leads.append(Lead(
                label=label,
                samples=values,
                sampling_rate=round(header.sampling_rate * spec.samples_per_frame),
                resolution=resolution,
                resolution_unit=unit,
                offset=offset,
                units="" if is_raw else unit,
                is_raw=is_raw,
                adc_resolution=spec.gain,
                adc_resolution_unit=adc_unit,
            ))

        if self._bad_checksums:
            self._warnings.append((
                f"WFDB checksum mismatch for signals {self._bad_checksums}",
                ChecksumWarning,
            ))
        if not header.sampling_rate_stated:
            # 250 Hz is the WFDB default when the header gives no frequency
            self._warnings.append((
                "WFDB header has no sampling frequency, using the WFDB default of 250 Hz",
                UserWarning,
            ))
        self._invalid_samples = invalid
        return leads

    def _fill_metadata(
        self, record: ECGRecord, header: _Header, specs: list[_SignalSpec], sig_len: int
    ) -> None:
        patient = PatientInfo()
        context: dict = {"interpretation_statements": [], "medications": []}
        self._parse_comments(header.comments, patient, context)
        patient.medications.extend(context["medications"])
        record.patient = patient
        if context["interpretation_statements"]:
            record.interpretation = Interpretation(
                statements=context["interpretation_statements"],
            )

        recording = RecordingInfo()
        if header.base_date and header.base_time:
            recording.date = datetime.combine(header.base_date.date(), header.base_time)
        fs = header.sampling_rate
        if sig_len > 0:
            recording.duration = timedelta(seconds=sig_len / fs)
        if context.get("technician"):
            recording.technician = context["technician"]
        if context.get("institution"):
            recording.device.institution = context["institution"]
        if context.get("device_model"):
            recording.device.model = context["device_model"]

        fmts = sorted(self._formats)
        bits = {s.adc_resolution for s in specs}
        zeros = {s.adc_zero for s in specs}
        recording.acquisition.signal = SignalCharacteristics(
            bits_per_sample=bits.pop() if len(bits) == 1 else None,
            signal_offset=(zeros.pop() or None) if len(zeros) == 1 else None,
            signal_signed=True,
            number_channels_allocated=header.num_signals,
            number_channels_valid=len(record.leads),
            data_encoding=",".join(f"format_{f}" for f in fmts),
            compression="none",
        )
        rates = {lead.sampling_rate for lead in record.leads}
        if len(rates) == 1:
            recording.acquisition.signal.sampling_rate = rates.pop()
        elif record.leads:
            recording.acquisition.signal.sampling_rate = round(fs)
        record.recording = recording

        version = "WFDB multi-segment" if header.num_segments is not None else "WFDB"
        if fmts:
            version += " format " + "/".join(str(f) for f in fmts)
        record.file_format = FileFormatInfo(version=version)

        raw = record.raw_metadata
        raw["filepath"] = str(self._path)
        raw["record_name"] = header.record_name
        raw["sampling_frequency"] = fs
        raw["sampling_frequency_stated"] = header.sampling_rate_stated
        if header.counter_frequency is not None:
            raw["counter_frequency"] = header.counter_frequency
        if header.base_counter is not None:
            raw["base_counter"] = header.base_counter
        if header.base_time and not header.base_date:
            raw["base_time"] = header.base_time.isoformat()
        if header.base_date and not header.base_time:
            raw["base_date"] = header.base_date.date().isoformat()
        if header.num_segments is not None:
            raw["segments"] = list(header.segments)
        raw["signal_specs"] = [vars(s).copy() for s in specs]
        raw["checksum_valid"] = all(self._checksums) if self._checksums else None
        if self._invalid_samples:
            raw["invalid_sample_counts"] = self._invalid_samples
        if header.comments:
            raw["comments"] = list(header.comments)

    def _parse_comments(
        self, comments: list[str], patient: PatientInfo, context: dict
    ) -> None:
        in_diagnoses = False
        for index, comment in enumerate(comments):
            if index == 0 and ":" not in comment:
                # MIT-BIH style "69 M 1085 1629 x1": age and sex first
                m = _MIT_AGE_SEX_RE.match(comment)
                if m:
                    if m.group(1) != "?":
                        patient.age = int(m.group(1))
                    patient.sex = {"M": "M", "F": "F"}.get(m.group(2), "U")
                    continue
            if in_diagnoses and comment:
                # LUDB lists one diagnosis per comment line after "<diagnoses>:"
                context["interpretation_statements"].append((comment, ""))
                continue
            if ":" not in comment:
                continue
            key, _, value = comment.partition(":")
            key = key.strip().strip("<>").strip().lower()
            value = value.strip()
            if key in ("diagnoses", "diagnosis") and not value:
                in_diagnoses = True
                continue
            self._parse_comment_field(key, value, patient, context)

    @staticmethod
    def _parse_comment_field(
        key: str, value: str, patient: PatientInfo, context: dict
    ) -> None:
        if not value:
            return
        if key in ("age", "patient age"):
            try:
                patient.age = round(float(value))
            except ValueError:
                pass
        elif key in ("sex", "gender"):
            v = value.upper()
            patient.sex = "M" if v in ("M", "MALE") else "F" if v in ("F", "FEMALE") else "U"
        elif key in ("id", "patient id"):
            patient.patient_id = value
        elif key in ("height", "patient height"):
            try:
                patient.height = float(value)
            except ValueError:
                pass
        elif key in ("weight", "patient weight"):
            try:
                patient.weight = float(value)
            except ValueError:
                pass
        elif key in ("dx", "diagnosis", "diagnoses"):
            context["interpretation_statements"].append((value, ""))
        elif key in ("drugs", "medications", "medication"):
            context["medications"].append(value)
        elif key in ("name", "patient name"):
            name_parts = value.split(None, 1)
            patient.first_name = name_parts[0]
            if len(name_parts) > 1:
                patient.last_name = name_parts[1]
        elif key in ("race", "ethnicity"):
            patient.race = value
        elif key in ("history", "clinical_history", "clinical history"):
            patient.clinical_history = value
        elif key == "technician":
            context.setdefault("technician", value)
        elif key in ("institution", "hospital"):
            context.setdefault("institution", value)
        elif key in ("device", "equipment"):
            context.setdefault("device_model", value)

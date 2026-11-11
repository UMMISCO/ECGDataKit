"""Cardionics Holter chain parser.

Cardionics recorders split a Holter recording into ISHNE Holter files of a
fixed length (chunks). The first chunk has the name of the ``.chn`` file
with the ``.ecg`` extension. Each time the recorder starts a new chunk, it
adds a record to the ``.chn`` file.

* Records of 276 bytes, little-endian, in recording order.
* Bytes 0 to 3: nominal chunk length in seconds (unsigned 32-bit).
* Bytes 4 to 7: time from the start of the recording to the start of the
  chunk, in seconds (unsigned 32-bit).
* Bytes 8 to 21: time the record was written, seven unsigned 16-bit
  values: day, month, year, hour, minute, second, millisecond.
* Bytes 22 to 275: file name of the chunk, ended by a NUL.
* The last record is named ``End``. Its time is the length of the whole
  recording. The bytes after its name are not data.

A chunk lasts from its start to the start of the next one, or to the
``End`` record for the last chunk. Each chunk is read with
:class:`ISHNEHolterParser`, and the leads of the chunks are joined in the
order of the chain. The samples of each chunk, and of all the chunks
together, must last as long as the chain says, within 1 s since the chain
stores whole seconds.

A chunk that cannot be read is a gap: its samples are NaN, for the time the
chain gives it, so the next samples keep their time. If the first chunk
cannot be read, the start of the recording is taken from the other chunks.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import re
import struct
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ecgdatakit.exceptions import CorruptedFileError, ECGDataKitError
from ecgdatakit.models import ECGFile, ECGRecord, FileFormatInfo
from ecgdatakit.parsing.parser import Parser

# Imported as a module so that parser discovery does not list ISHNEHolterParser twice
from ecgdatakit.parsing.parsers import ishne_holter

_EXT = ".chn"
_RECORD = 276
_NAME_OFFSET = 22
_END = "end"
_CHUNK_EXT = ".ecg"
_TOLERANCE_S = 2.0
_LENGTH_TOLERANCE_MS = 1000
# A chain record is written a few seconds after its chunk starts
_WRITE_TOLERANCE_S = 60.0

# Lead fields that must match between chunks for their samples to be joined
_LEAD_FIELDS = ("label", "sampling_rate", "resolution", "resolution_unit", "offset",
                "units", "is_raw", "adc_resolution", "adc_resolution_unit")
_PATIENT_FIELDS = ("patient_id", "first_name", "last_name", "birth_date", "sex")


@dataclass
class _Entry:
    """One record of the chain."""
    nominal_s: int
    elapsed_s: int
    written_at: datetime | None
    name: str


@dataclass
class _Segment:
    """One chunk of the recording, as the chain describes it."""
    name: str
    start_s: int
    end_s: int | None
    nominal_s: int | None
    written_at: datetime | None
    listed: bool


@dataclass
class _Chunk:
    """One chunk file. *record* is None if the file cannot be read."""
    segment: _Segment
    path: Path
    record: ECGRecord | None
    # Samples per lead in the joined leads
    samples: int
    # Why the file cannot be read, and the error raised if no chunk can be
    error: str = ""
    failure: ECGDataKitError | None = None


def _read_entry(data: bytes, offset: int) -> _Entry:
    nominal_s, elapsed_s = struct.unpack_from("<II", data, offset)
    day, month, year, hour, minute, second, ms = struct.unpack_from("<7H", data, offset + 8)
    try:
        written_at = datetime(year, month, day, hour, minute, second, ms * 1000)
    except ValueError:
        written_at = None
    name = data[offset + _NAME_OFFSET: offset + _RECORD].split(b"\x00", 1)[0]
    # Only the file name is used, a stored folder is ignored
    name = re.split(r"[\\/]", name.decode("latin-1").strip())[-1]
    return _Entry(nominal_s, elapsed_s, written_at, name)


def _rate(chunks: list[_Chunk]) -> int:
    """Sampling rate of the recording, from the first chunk that can be read."""
    return next(c.record for c in chunks if c.record is not None).leads[0].sampling_rate


def _checksum(chunk: _Chunk) -> bool | None:
    """True or False for a right or wrong header checksum, None if the header stores none."""
    return chunk.record.raw_metadata.get("checksum_valid")


def _difference(record: ECGRecord, first: _Chunk) -> str | None:
    """How *record* differs from the chunk *first*, as the end of an error message."""
    if len(record.leads) != len(first.record.leads):
        return (f" has {len(record.leads)} leads, "
                f"{first.segment.name} has {len(first.record.leads)}")
    for lead, ref in zip(record.leads, first.record.leads):
        for name in _LEAD_FIELDS:
            if getattr(lead, name) != getattr(ref, name):
                return (f": lead {ref.label} has {name} {getattr(lead, name)!r}, "
                        f"{first.segment.name} has {getattr(ref, name)!r}")
    for name in _PATIENT_FIELDS:
        if getattr(record.patient, name) != getattr(first.record.patient, name):
            return f" holds another patient ({name} differs from {first.segment.name})"
    return None


class CardionicsChainParser(Parser):
    """Parser for Cardionics ``.chn`` Holter chains."""

    FORMAT_NAME = "Cardionics Holter chain"
    FORMAT_DESCRIPTION = "Cardionics Holter recording split into ISHNE files (.chn)"
    FILE_EXTENSIONS = [_EXT]
    PRIORITY = 45

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        path = Path(file_path)
        if path.suffix.lower() != _EXT or not path.is_file() or len(header) < _RECORD:
            return False
        entry = _read_entry(header, 0)
        name = entry.name.lower()
        return entry.written_at is not None and (name == _END or name.endswith(_CHUNK_EXT))

    def parse(self, file_path: Path) -> ECGRecord:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        segments, end = self._read_chain(path)
        chunks = [self._read_chunk(path, s, self._locate(path, s.name)) for s in segments]
        unread = [c for c in chunks if c.record is None]
        if len(unread) == len(chunks):
            raise unread[0].failure
        for chunk in unread:
            chunk.failure = None
        reference = self._check_chunks(path, chunks)
        self._size_gaps(path, chunks, end)
        self._check_duration(path, chunks)
        start, source = self._recording_start(path, chunks, end)
        self._check_timeline(path, chunks, start)
        return self._join(path, chunks, end, start, source, reference)

    @staticmethod
    def _read_chain(path: Path) -> tuple[list[_Segment], _Entry | None]:
        data = path.read_bytes()
        if len(data) < _RECORD:
            raise CorruptedFileError(f"Cardionics chain: {path.name} holds no complete record")
        if len(data) % _RECORD:
            warnings.warn(
                f"Cardionics chain: {path.name} ends with {len(data) % _RECORD} bytes "
                "that do not form a whole record, ignored",
                stacklevel=3,
            )
        records = [_read_entry(data, offset)
                   for offset in range(0, len(data) - _RECORD + 1, _RECORD)]
        entries: list[_Entry] = []
        end = None
        for index, entry in enumerate(records):
            if entry.name.lower() == _END:
                end = entry
                if index + 1 < len(records):
                    warnings.warn(
                        f"Cardionics chain: {len(records) - index - 1} record(s) after "
                        f"the End record of {path.name} ignored",
                        stacklevel=3,
                    )
                break
            entries.append(entry)
        if end is None:
            warnings.warn(
                f"Cardionics chain: {path.name} has no End record, the recording "
                "may be incomplete",
                stacklevel=3,
            )
        for entry in entries:
            if not entry.name.lower().endswith(_CHUNK_EXT):
                raise CorruptedFileError(
                    f"Cardionics chain: {path.name} lists {entry.name!r}, not an ISHNE "
                    f"{_CHUNK_EXT} file"
                )

        # The first chunk is not listed: it has the name of the chain
        first = path.stem + _CHUNK_EXT
        names = [first.lower()] + [entry.name.lower() for entry in entries]
        if len(set(names)) != len(names):
            raise CorruptedFileError(f"Cardionics chain: {path.name} lists a chunk twice")
        starts = [0] + [entry.elapsed_s for entry in entries]
        bounds = starts + ([end.elapsed_s] if end is not None else [])
        # End at 0 s alone: a recording stopped within its first second
        if bounds != [0, 0] and any(b <= a for a, b in itertools.pairwise(bounds)):
            raise CorruptedFileError(
                f"Cardionics chain: times of {path.name} do not increase: {bounds[1:]}"
            )
        segments = [_Segment(first, 0, None, None, None, False)] + [
            _Segment(e.name, e.elapsed_s, None, e.nominal_s, e.written_at, True)
            for e in entries
        ]
        for segment, stop in zip(segments, bounds[1:]):
            segment.end_s = stop
        return segments, end

    @staticmethod
    def _locate(chain: Path, name: str) -> Path | None:
        exact = chain.parent / name
        if exact.is_file():
            return exact
        # The chain may not match the case of the file names on disk
        wanted = name.lower()
        return next((p for p in chain.parent.iterdir()
                     if p.name.lower() == wanted and p.is_file()), None)

    @staticmethod
    def _read_chunk(chain: Path, segment: _Segment, path: Path | None) -> _Chunk:
        """Read a chunk file. An unreadable file gives a chunk without record."""
        where = f"Cardionics chain: chunk {segment.name} of {chain.name}"
        if path is None:
            raise CorruptedFileError(f"{where} is missing")
        caught: list[warnings.WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                record = ishne_holter.ISHNEHolterParser().parse(path)
        except ECGDataKitError as e:
            failure = type(e)(f"{where}: {e}")
            # The cause is kept without its traceback, which holds memory
            failure.__cause__ = e.with_traceback(None)
            return _Chunk(segment, path, None, 0, str(e), failure)
        except (OSError, ImportError):
            raise
        except Exception as e:
            # Not a sign of a damaged file: raised, not turned into a gap
            raise CorruptedFileError(f"{where}: cannot decode: {type(e).__name__}: {e}") from e
        finally:
            # Warnings of a chunk name the chunk they come from
            for w in caught:
                warnings.warn(f"{where}: {w.message}", w.category, stacklevel=3)
        if not record.leads:
            raise CorruptedFileError(f"{where} holds no lead")
        return _Chunk(segment, path, record, len(record.leads[0].samples))

    @staticmethod
    def _check_chunks(chain: Path, chunks: list[_Chunk]) -> _Chunk:
        """Raise when a chunk does not continue the reference chunk, which is returned.

        The reference is the first readable chunk with a right header
        checksum, or the first readable chunk if none has one. A chunk that
        differs from it and has a wrong checksum has a damaged header: it
        becomes unreadable instead of raising.
        """
        readable = [c for c in chunks if c.record is not None]
        checked = [c for c in readable if _checksum(c) is True]
        first = checked[0] if checked else readable[0]
        for chunk in readable:
            difference = _difference(chunk.record, first) if chunk is not first else None
            if difference is None:
                continue
            if checked and _checksum(chunk) is False:
                chunk.error = f"damaged header (wrong checksum), {difference.lstrip(': ')}"
                chunk.record, chunk.samples = None, 0
                continue
            raise CorruptedFileError(
                f"Cardionics chain: chunk {chunk.segment.name} of {chain.name}{difference}")
        return first

    @staticmethod
    def _check_duration(chain: Path, chunks: list[_Chunk]) -> None:
        """Raise when the samples do not last as long as the chain says.

        Each chunk is checked against its start and end in the chain, then all
        the chunks against the End record. Without an End record, the last
        chunk and the total are not checked. An unreadable chunk is not
        checked. The total counts it for its time in the chain.
        """
        rate = _rate(chunks)
        wrong: list[str] = []
        for chunk in chunks:
            segment = chunk.segment
            if segment.end_s is None or chunk.record is None:
                continue
            held_ms = round(1000 * chunk.samples / rate)
            stated_ms = 1000 * (segment.end_s - segment.start_s)
            if abs(held_ms - stated_ms) > _LENGTH_TOLERANCE_MS:
                wrong.append(f"{segment.name} lasts {held_ms} ms, chain says {stated_ms} ms")
        if wrong:
            raise CorruptedFileError(
                f"Cardionics chain: chunk lengths of {chain.name} do not match its "
                f"times: {', '.join(wrong)}"
            )
        if chunks[-1].segment.end_s is None:
            return
        held_ms = round(1000 * sum(c.samples for c in chunks) / rate)
        stated_ms = 1000 * chunks[-1].segment.end_s
        if abs(held_ms - stated_ms) > _LENGTH_TOLERANCE_MS:
            raise CorruptedFileError(
                f"Cardionics chain: the samples of {chain.name} last {held_ms} ms, "
                f"its chunks last {stated_ms} ms"
            )

    @staticmethod
    def _size_gaps(chain: Path, chunks: list[_Chunk], end: _Entry | None) -> None:
        """Set the length of each unreadable chunk from its time in the chain.

        Raise when that time is longer than a chunk can last: the chain is
        then wrong, and the gap would have any length.
        """
        rate = _rate(chunks)
        # Chunk length of the recorder. The first chunk has no record, so no length of its own
        nominal = next((n for n in [*(c.segment.nominal_s for c in chunks),
                                    end.nominal_s if end is not None else None] if n), None)
        longest = math.ceil(max(c.samples for c in chunks if c.record is not None) / rate)
        for chunk in chunks:
            if chunk.record is not None:
                continue
            segment = chunk.segment
            where = f"Cardionics chain: chunk {segment.name} of {chain.name}"
            if segment.end_s is None:
                # Last chunk without End record: its length is unknown
                warnings.warn(
                    f"{where} cannot be read and its length is unknown, it is left out: "
                    f"{chunk.error}",
                    stacklevel=3,
                )
                continue
            seconds = segment.end_s - segment.start_s
            limit = segment.nominal_s or nominal or longest
            if seconds > limit + _LENGTH_TOLERANCE_MS // 1000:
                raise CorruptedFileError(
                    f"{where} cannot be read and the chain gives it {seconds} s, more than "
                    f"the {limit} s of a chunk: {chunk.error}"
                )
            chunk.samples = round(seconds * rate)
            warnings.warn(
                f"{where} cannot be read, its {seconds} s are filled with NaN: "
                f"{chunk.error}",
                stacklevel=3,
            )

    @staticmethod
    def _recording_start(chain: Path, chunks: list[_Chunk],
                         end: _Entry | None) -> tuple[datetime | None, _Chunk | None]:
        """Start of the recording, and the chunk it is taken from if not the first.

        If the first chunk cannot be read, the start comes from the other
        headers. A header holds the start of the recording or of its chunk.
        Two headers that agree tell which. Otherwise the chain decides: the
        write time of a record minus its elapsed time is close to the start.
        A start that is not close to it is not kept.
        """
        if chunks[0].record is not None:
            return chunks[0].record.recording.date, None
        rate = _rate(chunks)
        # Chunk, header time, and header time minus the time before the chunk
        dated: list[tuple[_Chunk, datetime, datetime]] = []
        before = 0
        for chunk in chunks:
            date = chunk.record.recording.date if chunk.record is not None else None
            if date is not None:
                dated.append((chunk, date, date - timedelta(seconds=before / rate)))
            before += chunk.samples
        unknown = (f"Cardionics chain: the first chunk of {chain.name} cannot be read and "
                   "the other chunks do not tell the start of the recording, it is not known")
        if not dated:
            warnings.warn(unknown, stacklevel=3)
            return None, None

        def agree(times: list[datetime]) -> bool:
            return all(abs((t - times[0]).total_seconds()) <= _TOLERANCE_S for t in times)

        source, same, own = dated[0]
        if len(dated) > 1 and agree([d[1] for d in dated]):
            start = same
        elif len(dated) > 1 and agree([d[2] for d in dated]):
            start = own
        else:
            entries = [(c.segment.written_at, c.segment.start_s) for c in chunks]
            if end is not None:
                entries.append((end.written_at, end.elapsed_s))
            written = [at - timedelta(seconds=s) for at, s in entries if at is not None]
            away = [abs((t - written[0]).total_seconds()) if written else math.inf
                    for t in (same, own)]
            if min(away) > _WRITE_TOLERANCE_S:
                warnings.warn(unknown, stacklevel=3)
                return None, None
            start = same if away[0] <= away[1] else own
        warnings.warn(
            f"Cardionics chain: the first chunk of {chain.name} cannot be read, the start "
            f"of the recording ({start:%Y-%m-%d %H:%M:%S}) is taken from chunk "
            f"{source.segment.name}",
            stacklevel=3,
        )
        return start, source

    @staticmethod
    def _check_timeline(chain: Path, chunks: list[_Chunk], origin: datetime | None) -> None:
        """Warn when the chunk start times do not match the recording.

        A chunk header holds the start of the recording or the start of the
        chunk, both are accepted.
        """
        rate = _rate(chunks)
        starts: list[str] = []
        before = 0
        for chunk in chunks:
            segment = chunk.segment
            date = chunk.record.recording.date if chunk.record is not None else None
            if origin is not None and date is not None:
                follows = origin + timedelta(seconds=before / rate)
                if all(abs((date - t).total_seconds()) > _TOLERANCE_S for t in (origin, follows)):
                    starts.append(f"{segment.name} starts at {date:%H:%M:%S}, expected "
                                  f"{origin:%H:%M:%S} or {follows:%H:%M:%S}")
            before += chunk.samples
        if starts:
            warnings.warn(
                f"Cardionics chain: chunk start times of {chain.name} do not match the "
                f"recording: {', '.join(starts)}",
                stacklevel=3,
            )

    @staticmethod
    def _join(chain: Path, chunks: list[_Chunk], end: _Entry | None, origin: datetime | None,
              source: _Chunk | None, reference: _Chunk) -> ECGRecord:
        # Patient, device and lead settings come from the reference chunk
        first = reference.record
        templates: list = list(first.leads)
        # None for an unreadable chunk, replaced by NaN below
        parts = [[c.record.leads[i].samples if c.record is not None else None for c in chunks]
                 for i in range(len(templates))]
        for c in chunks:
            if c.record is not None:
                c.record.leads = []
        # Lead by lead, so that a long recording is held at most once plus one lead
        leads = []
        for i in range(len(templates)):
            template, templates[i] = templates[i], None
            samples = np.concatenate([np.full(c.samples, np.nan) if part is None else part
                                      for c, part in zip(chunks, parts[i])])
            leads.append(dataclasses.replace(template, samples=samples))
            parts[i] = []

        rate = leads[0].sampling_rate
        total = sum(c.samples for c in chunks)
        record = ECGRecord(source_format="cardionics_chain")
        record.patient = first.patient
        # Copied, so the chunk keeps its own header time
        record.recording = dataclasses.replace(
            first.recording, date=origin, duration=timedelta(seconds=total / rate))
        record.file_format = FileFormatInfo()
        record.leads = leads
        record.representative_beats = []
        record.annotations = dict(first.annotations)

        chunk_meta = []
        position = 0
        for c in chunks:
            segment = c.segment
            end_s: float | None = segment.end_s
            if end_s is None and c.record is not None:
                end_s = segment.start_s + c.samples / rate
            record.ecg_files.append(ECGFile(
                name=segment.name,
                start=origin + timedelta(seconds=segment.start_s) if origin else None,
                end=origin + timedelta(seconds=end_s) if origin and end_s is not None else None,
                samples=c.samples,
                error=c.error,
            ))
            chunk_meta.append({
                "name": segment.name,
                "filepath": str(c.path),
                "listed": segment.listed,
                "first_sample": position,
                "samples": c.samples,
                "error": c.error or None,
                "start": c.record.recording.date if c.record is not None else None,
                "start_s": segment.start_s,
                "end_s": segment.end_s,
                "nominal_s": segment.nominal_s,
                "written_at": segment.written_at,
                "annotations": c.record.annotations if c.record is not None else {},
                "ishne": c.record.raw_metadata if c.record is not None else {},
            })
            position += c.samples
        raw = record.raw_metadata
        raw["filepath"] = str(chain)
        raw["chunks"] = chunk_meta
        raw["start_inferred_from"] = source.segment.name if source is not None else None
        raw["end_record"] = end is not None
        raw["end_elapsed_s"] = end.elapsed_s if end is not None else None
        raw["end_written_at"] = end.written_at if end is not None else None
        return record

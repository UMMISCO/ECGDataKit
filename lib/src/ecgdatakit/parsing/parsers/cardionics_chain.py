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
"""

from __future__ import annotations

import dataclasses
import itertools
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
    segment: _Segment
    path: Path
    record: ECGRecord
    samples: int


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
        self._check_chunks(path, chunks)
        self._check_duration(path, chunks)
        self._check_timeline(path, chunks)
        return self._join(path, chunks, end)

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
        if any(b <= a for a, b in itertools.pairwise(bounds)):
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
        where = f"Cardionics chain: chunk {segment.name} of {chain.name}"
        if path is None:
            raise CorruptedFileError(f"{where} is missing")
        caught: list[warnings.WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                record = ishne_holter.ISHNEHolterParser().parse(path)
        except ECGDataKitError as e:
            raise type(e)(f"{where}: {e}") from e
        except (OSError, ImportError):
            raise
        except Exception as e:
            raise CorruptedFileError(f"{where}: cannot decode: {type(e).__name__}: {e}") from e
        finally:
            # Warnings of a chunk name the chunk they come from
            for w in caught:
                warnings.warn(f"{where}: {w.message}", w.category, stacklevel=3)
        if not record.leads:
            raise CorruptedFileError(f"{where} holds no lead")
        return _Chunk(segment, path, record, len(record.leads[0].samples))

    @staticmethod
    def _check_chunks(chain: Path, chunks: list[_Chunk]) -> None:
        """Raise when a chunk does not continue the first one."""
        first = chunks[0]
        for chunk in chunks[1:]:
            record = chunk.record
            where = f"Cardionics chain: chunk {chunk.segment.name} of {chain.name}"
            if len(record.leads) != len(first.record.leads):
                raise CorruptedFileError(
                    f"{where} has {len(record.leads)} leads, "
                    f"{first.segment.name} has {len(first.record.leads)}"
                )
            for lead, ref in zip(record.leads, first.record.leads):
                for name in _LEAD_FIELDS:
                    if getattr(lead, name) != getattr(ref, name):
                        raise CorruptedFileError(
                            f"{where}: lead {ref.label} has {name} {getattr(lead, name)!r}, "
                            f"{first.segment.name} has {getattr(ref, name)!r}"
                        )
            for name in _PATIENT_FIELDS:
                if getattr(record.patient, name) != getattr(first.record.patient, name):
                    raise CorruptedFileError(
                        f"{where} holds another patient "
                        f"({name} differs from {first.segment.name})"
                    )

    @staticmethod
    def _check_duration(chain: Path, chunks: list[_Chunk]) -> None:
        """Raise when the samples do not last as long as the chain says.

        Each chunk is checked against its start and end in the chain, then all
        the chunks against the End record. Without an End record, the last
        chunk and the total are not checked.
        """
        rate = chunks[0].record.leads[0].sampling_rate
        wrong: list[str] = []
        for chunk in chunks:
            segment = chunk.segment
            if segment.end_s is None:
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
    def _check_timeline(chain: Path, chunks: list[_Chunk]) -> None:
        """Warn when the chunk start times do not match the recording.

        A chunk header holds the start of the recording or the start of the
        chunk, both are accepted.
        """
        rate = chunks[0].record.leads[0].sampling_rate
        origin = chunks[0].record.recording.date
        starts: list[str] = []
        before = 0
        for chunk in chunks:
            segment = chunk.segment
            date = chunk.record.recording.date
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
    def _join(chain: Path, chunks: list[_Chunk], end: _Entry | None) -> ECGRecord:
        first = chunks[0].record
        templates: list = list(first.leads)
        parts = [[c.record.leads[i].samples for c in chunks] for i in range(len(templates))]
        for c in chunks:
            c.record.leads = []
        # Lead by lead, so that a long recording is held at most once plus one lead
        leads = []
        for i in range(len(templates)):
            template, templates[i] = templates[i], None
            leads.append(dataclasses.replace(template, samples=np.concatenate(parts[i])))
            parts[i] = []

        rate = leads[0].sampling_rate
        total = sum(c.samples for c in chunks)
        record = ECGRecord(source_format="cardionics_chain")
        record.patient = first.patient
        record.recording = first.recording
        record.recording.duration = timedelta(seconds=total / rate)
        record.file_format = FileFormatInfo()
        record.leads = leads
        record.representative_beats = []
        record.annotations = dict(first.annotations)

        origin = record.recording.date
        chunk_meta = []
        position = 0
        for c in chunks:
            segment = c.segment
            end_s = segment.end_s if segment.end_s is not None else segment.start_s + c.samples / rate
            record.ecg_files.append(ECGFile(
                name=segment.name,
                start=origin + timedelta(seconds=segment.start_s) if origin else None,
                end=origin + timedelta(seconds=end_s) if origin else None,
                samples=c.samples,
            ))
            chunk_meta.append({
                "name": segment.name,
                "filepath": str(c.path),
                "listed": segment.listed,
                "first_sample": position,
                "samples": c.samples,
                "start": c.record.recording.date,
                "start_s": segment.start_s,
                "end_s": segment.end_s,
                "nominal_s": segment.nominal_s,
                "written_at": segment.written_at,
                "annotations": c.record.annotations,
                "ishne": c.record.raw_metadata,
            })
            position += c.samples
        raw = record.raw_metadata
        raw["filepath"] = str(chain)
        raw["chunks"] = chunk_meta
        raw["end_record"] = end is not None
        raw["end_elapsed_s"] = end.elapsed_s if end is not None else None
        raw["end_written_at"] = end.written_at if end is not None else None
        return record

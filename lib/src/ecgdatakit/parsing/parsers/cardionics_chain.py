"""Cardionics Holter chain parser.

Cardionics recorders split a Holter recording into ISHNE Holter files of a
fixed length (chunks) and list them in a ``.chn`` file.

* Records of 276 bytes, little-endian, one per chunk in recording order.
* Bytes 0 to 3: nominal chunk length in seconds (unsigned 32-bit).
* Bytes 4 to 7: time from the start of the recording to the end of the
  chunk, in seconds (unsigned 32-bit).
* Bytes 8 to 21: time the record was written, seven unsigned 16-bit
  values: day, month, year, hour, minute, second, millisecond.
* Bytes 22 to 275: file name of the chunk, ended by a NUL.
* The last record is named ``End``. Its elapsed time is the length of the
  whole recording. The bytes after its name are not data.

The last chunk is not named in the chain, the ``End`` record takes its
place. When the recording goes on after the last listed chunk, the last
chunk is the file numbered after the last listed one (``rec_3.ecg``
after ``rec_2.ecg``). It is read when it exists, and it is required when
the listed chunks are more than 2 s shorter than the recording.

Each chunk is read with :class:`ISHNEHolterParser`. The leads of the
chunks are joined in the order of the chain.
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
from ecgdatakit.models import ECGRecord, FileFormatInfo
from ecgdatakit.parsing.parser import Parser

# Imported as a module so that parser discovery does not list ISHNEHolterParser twice
from ecgdatakit.parsing.parsers import ishne_holter

_EXT = ".chn"
_RECORD = 276
_NAME_OFFSET = 22
_END = "end"
_CHUNK_EXT = ".ecg"
_TOLERANCE_S = 2.0
_NUMBERED = re.compile(r"^(?P<stem>.*\D)(?P<number>\d+)(?P<ext>\.ecg)$", re.IGNORECASE)

# Lead fields that must match between chunks for their samples to be joined
_LEAD_FIELDS = ("label", "sampling_rate", "resolution", "resolution_unit", "offset",
                "units", "is_raw", "adc_resolution", "adc_resolution_unit")
_PATIENT_FIELDS = ("patient_id", "first_name", "last_name", "birth_date", "sex")


@dataclass
class _Entry:
    nominal_s: int
    elapsed_s: int
    written_at: datetime | None
    name: str


@dataclass
class _Chunk:
    entry: _Entry
    path: Path
    record: ECGRecord
    listed: bool
    samples: int = 0


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
        entries, end = self._read_chain(path)
        chunks = [self._read_chunk(path, entry, self._locate(path, entry.name), True)
                  for entry in entries]
        self._check_chunks(path, chunks)
        rate = chunks[0].record.leads[0].sampling_rate
        if end is not None and end.elapsed_s > entries[-1].elapsed_s:
            held = sum(c.samples for c in chunks) / rate
            last = self._last_chunk(path, entries[-1], end, held)
            if last is not None:
                chunks.append(last)
                self._check_chunks(path, chunks)
        self._check_timeline(path, chunks, rate)
        return self._join(path, chunks, end)

    @staticmethod
    def _read_chain(path: Path) -> tuple[list[_Entry], _Entry | None]:
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
        if not entries:
            raise CorruptedFileError(f"Cardionics chain: {path.name} lists no chunk")
        for entry in entries:
            if not entry.name.lower().endswith(_CHUNK_EXT):
                raise CorruptedFileError(
                    f"Cardionics chain: {path.name} lists {entry.name!r}, not an ISHNE "
                    f"{_CHUNK_EXT} file"
                )
        names = [entry.name.lower() for entry in entries]
        if len(set(names)) != len(names):
            raise CorruptedFileError(f"Cardionics chain: {path.name} lists a chunk twice")
        elapsed = [0] + [entry.elapsed_s for entry in entries]
        # The recording may stop when the last listed chunk ends
        if any(b <= a for a, b in itertools.pairwise(elapsed)) or (
                end is not None and end.elapsed_s < elapsed[-1]):
            stored = elapsed[1:] + ([end.elapsed_s] if end is not None else [])
            raise CorruptedFileError(
                f"Cardionics chain: elapsed times of {path.name} do not increase: {stored}"
            )
        return entries, end

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
    def _read_chunk(chain: Path, entry: _Entry, path: Path | None, listed: bool) -> _Chunk:
        if path is None:
            raise CorruptedFileError(
                f"Cardionics chain: chunk {entry.name} of {chain.name} is missing"
            )
        where = f"Cardionics chain: chunk {entry.name} of {chain.name}"
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
            raise CorruptedFileError(
                f"Cardionics chain: chunk {entry.name} of {chain.name} holds no lead"
            )
        return _Chunk(entry, path, record, listed, len(record.leads[0].samples))

    def _last_chunk(self, chain: Path, last: _Entry, end: _Entry,
                    held: float) -> _Chunk | None:
        """The chunk after the last listed one, which the End record replaces.

        ``None`` when it does not exist and the listed chunks already hold the
        recording, within the tolerance.
        """
        required = end.elapsed_s - held > _TOLERANCE_S
        match = _NUMBERED.match(last.name)
        if match is None:
            if required:
                raise CorruptedFileError(
                    f"Cardionics chain: {chain.name} ends at {end.elapsed_s} s, its listed "
                    f"chunks hold {held:.1f} s, and the chunk after {last.name} cannot be "
                    "named from it"
                )
            return None
        digits = match["number"]
        name = f"{match['stem']}{str(int(digits) + 1).zfill(len(digits))}{match['ext']}"
        path = self._locate(chain, name)
        if path is None:
            if required:
                raise CorruptedFileError(
                    f"Cardionics chain: {chain.name} ends at {end.elapsed_s} s, its listed "
                    f"chunks hold {held:.1f} s, and the last chunk {name} is missing"
                )
            return None
        entry = _Entry(end.nominal_s, end.elapsed_s, end.written_at, name)
        return self._read_chunk(chain, entry, path, False)

    @staticmethod
    def _check_chunks(chain: Path, chunks: list[_Chunk]) -> None:
        """Raise when a chunk does not continue the first one."""
        first = chunks[0].record
        for chunk in chunks[1:]:
            record = chunk.record
            if len(record.leads) != len(first.leads):
                raise CorruptedFileError(
                    f"Cardionics chain: chunk {chunk.entry.name} of {chain.name} has "
                    f"{len(record.leads)} leads, {chunks[0].entry.name} has {len(first.leads)}"
                )
            for lead, ref in zip(record.leads, first.leads):
                for name in _LEAD_FIELDS:
                    if getattr(lead, name) != getattr(ref, name):
                        raise CorruptedFileError(
                            f"Cardionics chain: lead {ref.label} of chunk {chunk.entry.name} "
                            f"of {chain.name} has {name} {getattr(lead, name)!r}, "
                            f"{chunks[0].entry.name} has {getattr(ref, name)!r}"
                        )
            for name in _PATIENT_FIELDS:
                if getattr(record.patient, name) != getattr(first.patient, name):
                    raise CorruptedFileError(
                        f"Cardionics chain: chunk {chunk.entry.name} of {chain.name} holds "
                        f"another patient ({name} differs from {chunks[0].entry.name})"
                    )

    @staticmethod
    def _check_timeline(chain: Path, chunks: list[_Chunk], rate: int) -> None:
        """Warn when the chunk lengths or start times do not match the chain."""
        start = chunks[0].record.recording.date
        lengths: list[str] = []
        starts: list[str] = []
        before = 0
        for chunk in chunks:
            begin_s = before / rate
            before += chunk.samples
            end_s = before / rate
            if abs(end_s - chunk.entry.elapsed_s) > _TOLERANCE_S:
                lengths.append(f"{chunk.entry.name} ends at {end_s:.1f} s, "
                               f"chain says {chunk.entry.elapsed_s} s")
            date = chunk.record.recording.date
            if start is not None and date is not None:
                expected = start + timedelta(seconds=begin_s)
                if abs((date - expected).total_seconds()) > _TOLERANCE_S:
                    starts.append(f"{chunk.entry.name} starts at {date:%H:%M:%S}, "
                                  f"expected {expected:%H:%M:%S}")
        if lengths:
            warnings.warn(
                f"Cardionics chain: chunk lengths of {chain.name} do not match its "
                f"elapsed times: {', '.join(lengths)}",
                stacklevel=3,
            )
        if starts:
            warnings.warn(
                f"Cardionics chain: chunk start times of {chain.name} do not follow "
                f"the samples of the previous chunks: {', '.join(starts)}",
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

        chunk_meta = []
        position = 0
        for c in chunks:
            chunk_meta.append({
                "name": c.entry.name,
                "filepath": str(c.path),
                "listed": c.listed,
                "first_sample": position,
                "samples": c.samples,
                "start": c.record.recording.date,
                "elapsed_s": c.entry.elapsed_s,
                "nominal_s": c.entry.nominal_s,
                "written_at": c.entry.written_at,
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

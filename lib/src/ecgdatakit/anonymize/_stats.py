"""Counts of a dataset before and after anonymization.

Both sides are counted from disk: the raw files, and the files of the
``ANONYMIZED`` folder, each read again. Every line has a check that must
hold after anonymization.
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import NamedTuple

from ecgdatakit.anonymize._identity import looks_like_date, matches_folder

STATS_NAME = "anonymization_stats.csv"
"""Stats file written next to the catalog after each run."""


class FileRecord(NamedTuple):
    """One ECG file found on disk."""

    path: str
    """Path relative to the counted folder."""
    patient_folder: str
    """Its patient folder ("" when patients are not grouped or the file is
    outside a patient folder)."""
    patient_ids: tuple[str, ...]
    ecg_ids: tuple[str, ...]
    sha256: str


class Stat(NamedTuple):
    what: str
    raw: int
    anonymized: int
    ok: bool
    check: str
    """What must hold for the line to be OK."""


def _parents(records: list[FileRecord]) -> set[str]:
    return {PurePosixPath(r.path).parent.as_posix() for r in records}


def _duplicates(records: list[FileRecord]) -> int:
    """Files with the same content as another file (each extra copy counted)."""
    return sum(n - 1 for n in Counter(r.sha256 for r in records).values())


def _ids(records: list[FileRecord]) -> set[str]:
    """Patient ID values of *records*, without dates (kept in the field,
    they do not name a patient)."""
    return {v for r in records for v in r.patient_ids if not looks_like_date(v)}


def _not_folder_raw(r: FileRecord) -> bool:
    folder = PurePosixPath(r.patient_folder).name
    return any(looks_like_date(v) or not matches_folder(v, folder) for v in r.patient_ids)


def _not_folder_anonymized(records: list[FileRecord], renamed: bool) -> int:
    """Anonymized files whose patient ID is not the one of their patient
    folder: the folder name when folders are renamed to the pseudonym,
    otherwise the ID most files of the folder hold. Dates, kept in the
    patient ID field, are left aside."""
    by_folder: dict[str, list[FileRecord]] = {}
    for r in records:
        if r.patient_folder:
            by_folder.setdefault(r.patient_folder, []).append(r)
    count = 0
    for folder, files in by_folder.items():
        if renamed:
            expected = PurePosixPath(folder).name
        else:
            values = Counter(v for r in files for v in r.patient_ids
                             if not looks_like_date(v))
            expected = values.most_common(1)[0][0] if values else ""
        count += sum(1 for r in files
                     if any(v != expected and not looks_like_date(v) for v in r.patient_ids))
    return count


def compare(raw: list[FileRecord], anon: list[FileRecord], grouped: bool,
            renamed: bool) -> list[Stat]:
    """Lines of the stats, from the raw and the anonymized files."""
    raw_in = [r for r in raw if r.patient_folder]
    anon_in = [r for r in anon if r.patient_folder]
    raw_folders = len({r.patient_folder for r in raw_in})
    anon_folders = len({r.patient_folder for r in anon_in})
    raw_ids = len(_ids(raw))
    anon_ids = len(_ids(anon))
    raw_with_id = sum(1 for r in raw if r.patient_ids)
    anon_with_id = sum(1 for r in anon if r.patient_ids)
    raw_not = sum(1 for r in raw_in if _not_folder_raw(r)) if grouped else 0
    anon_not = _not_folder_anonymized(anon, renamed) if grouped else 0
    anon_ecg = sum(1 for r in anon if r.ecg_ids)
    anon_ecg_ids = len({v for r in anon for v in r.ecg_ids})
    if grouped:
        # One ID per patient folder (folders whose files only hold dates
        # have none), and no ID shared by two folders
        per_folder: dict[str, set[str]] = {}
        for r in anon_in:
            per_folder.setdefault(r.patient_folder, set()).update(_ids([r]))
        one_each = all(len(v) <= 1 for v in per_folder.values())
        with_id = sum(1 for v in per_folder.values() if v)
        ids_line = Stat("distinct patient IDs in the files, dates aside", raw_ids, anon_ids,
                        one_each and len(_ids(anon_in)) == with_id,
                        "anonymized: one per patient folder")
    else:
        ids_line = Stat("distinct patient IDs in the files, dates aside", raw_ids, anon_ids,
                        raw_ids == anon_ids, "same (not replaced without grouping)")
    folders_raw, folders_anon = len(_parents(raw)), len(_parents(anon))
    return [
        Stat("folders holding ECG files, sub-folders included", folders_raw, folders_anon,
             folders_raw == folders_anon, "same"),
        Stat("patient folders", raw_folders, anon_folders, raw_folders == anon_folders, "same"),
        Stat("ECG files", len(raw), len(anon), len(raw) == len(anon), "same"),
        Stat("ECG files with a patient ID", raw_with_id, anon_with_id,
             raw_with_id == anon_with_id, "same"),
        Stat("ECG files whose patient ID is not their patient folder's", raw_not, anon_not,
             anon_not == 0, "anonymized: 0"),
        ids_line,
        Stat("duplicate ECG files (same content as another)", _duplicates(raw),
             _duplicates(anon), _duplicates(anon) == 0, "anonymized: 0"),
        Stat("distinct ECG IDs", len({v for r in raw for v in r.ecg_ids}), anon_ecg_ids,
             anon_ecg_ids == anon_ecg, "anonymized: one per ECG file"),
    ]


def format_stats(name: str, stats: list) -> str:
    stats = [Stat(*s) for s in stats]
    width = max(len(s.what) for s in stats)
    lines = [f"{name}", f"  {'':{width}}  {'raw':>8}  {'anonymized':>10}  check"]
    for s in stats:
        mark = "OK" if s.ok else "CHECK"
        lines.append(f"  {s.what:{width}}  {s.raw:>8}  {s.anonymized:>10}  {mark:5}  {s.check}")
    return "\n".join(lines)


def write_stats(path: Path, stats: list[Stat]) -> None:
    """Write *stats* as a CSV (count, raw, anonymized, ok, check)."""
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["count", "raw", "anonymized", "ok", "check"])
        for s in stats:
            writer.writerow([s.what, s.raw, s.anonymized, "TRUE" if s.ok else "FALSE", s.check])
    os.replace(temp, path)

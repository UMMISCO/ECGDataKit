"""Counts of a dataset before and after anonymization, read from its catalog."""

from __future__ import annotations

import csv
import os
from pathlib import Path, PurePosixPath

from ecgdatakit.anonymize._catalog import ACTIVE, SEPARATOR, Catalog

STATS_NAME = "anonymization_stats.csv"
"""Stats file written next to the catalog after each run."""
from ecgdatakit.anonymize._identity import looks_like_date, matches_folder


def _parents(paths) -> set[str]:
    return {PurePosixPath(p).parent.as_posix() for p in paths if p}


def _values(text: str) -> list[str]:
    return [v for v in text.split(SEPARATOR) if v.strip()]


def _kept(row: dict[str, str]) -> list[str]:
    """Patient ID values of the file kept as is: not the patient folder name,
    or a date. Without patient folder grouping every value is kept."""
    if not row["patient_folder"] or row["patient_folder"].startswith("file:"):
        return _values(row["file_patient_id"])
    folder = PurePosixPath(row["patient_folder"]).name
    return [v for v in _values(row["file_patient_id"])
            if looks_like_date(v) or not matches_folder(v, folder)]


def _written(row: dict[str, str]) -> str:
    """What the anonymized file holds in its patient ID field."""
    kept = _kept(row)
    return SEPARATOR.join(dict.fromkeys(
        v if v in kept else row["anonymized_patient_id"] for v in _values(row["file_patient_id"])))


def catalog_stats(catalog: Catalog) -> list[tuple[str, int, int]]:
    """``(what, raw, anonymized)`` counts of the catalog.

    The raw side counts every raw file still on disk (failed ones too), the
    anonymized side the files whose anonymized copy exists.
    """
    with catalog.lock:
        rows = list(catalog.rows.values())
    raw = [r for r in rows if r["status"] != "deleted"]
    done = [r for r in rows if r["status"] in ACTIVE]
    in_folder = [r for r in raw if r["patient_folder"] and not r["patient_folder"].startswith("file:")]
    folder_paths = {r["raw_path"] for r in in_folder}
    done_in_folder = [r for r in done if r["raw_path"] in folder_paths]
    return [
        ("folders with ECG files", len(_parents(r["raw_path"] for r in raw)),
         len(_parents(r["anonymized_path"] for r in done))),
        ("patient folders", len({r["patient_folder"] for r in in_folder}),
         len({r["anonymized_patient_id"] for r in done_in_folder})),
        ("patients", len({r["patient_folder"] or "file:" + r["raw_path"] for r in raw}),
         len({r["anonymized_patient_id"] for r in done})),
        ("files with a patient ID", sum(1 for r in raw if r["file_patient_id"]),
         sum(1 for r in done if r["file_patient_id"])),
        ("different patient IDs", len({r["file_patient_id"] for r in raw if r["file_patient_id"]}),
         len({_written(r) for r in done if r["file_patient_id"]})),
        ("files whose patient ID is not the folder name",
         sum(1 for r in in_folder if _kept(r)),
         sum(1 for r in done_in_folder if "patient ID field" in r["risk_reason"])),
        ("ECG files", len(raw), len(done)),
        ("ECG IDs", len({r["original_ecg_id"] for r in raw if r["original_ecg_id"]}),
         len({r["anonymized_ecg_id"] for r in done if r["original_ecg_id"]})),
    ]


def format_stats(name: str, stats: list[tuple[str, int, int]]) -> str:
    width = max(len(s[0]) for s in stats)
    lines = [f"{name}", f"  {'':{width}}  {'raw':>10}  {'anonymized':>10}"]
    for what, before, after in stats:
        mark = "" if before == after else "  differs"
        lines.append(f"  {what:{width}}  {before:>10}  {after:>10}{mark}")
    return "\n".join(lines)


def write_stats(path: Path, stats: list[tuple[str, int, int]]) -> None:
    """Write *stats* as a CSV (count, raw, anonymized, same)."""
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["count", "raw", "anonymized", "same"])
        for what, before, after in stats:
            writer.writerow([what, before, after, "TRUE" if before == after else "FALSE"])
    os.replace(temp, path)

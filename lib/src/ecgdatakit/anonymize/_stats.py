"""Counts of a dataset before and after anonymization, read from its catalog."""

from __future__ import annotations

from pathlib import PurePosixPath

from ecgdatakit.anonymize._catalog import ACTIVE, SEPARATOR, Catalog
from ecgdatakit.anonymize._engine import _matches_folder
from ecgdatakit.anonymize._identity import looks_like_date


def _parents(paths) -> set[str]:
    return {PurePosixPath(p).parent.as_posix() for p in paths if p}


def _values(text: str) -> list[str]:
    return [v for v in text.split(SEPARATOR) if v.strip()]


def _differs_from_folder(row: dict[str, str]) -> bool:
    """A patient ID value of the raw file is not the patient folder name."""
    folder = PurePosixPath(row["patient_folder"]).name
    return any(looks_like_date(v) or not _matches_folder(v, folder)
               for v in _values(row["file_patient_id"]))


def _not_pseudonym(row: dict[str, str]) -> bool:
    """The anonymized file holds a patient ID other than the pseudonym."""
    return any(v != row["patient_code"] for v in _values(row["anonymized_patient_id"]))


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
         len({r["patient_code"] for r in done_in_folder})),
        ("patients", len({r["patient_folder"] or "file:" + r["raw_path"] for r in raw}),
         len({r["patient_code"] for r in done})),
        ("files with a patient ID", sum(1 for r in raw if _values(r["file_patient_id"])),
         sum(1 for r in done if _values(r["anonymized_patient_id"]))),
        ("different patient IDs", len({r["file_patient_id"] for r in raw if r["file_patient_id"]}),
         len({r["anonymized_patient_id"] for r in done if r["anonymized_patient_id"]})),
        ("files whose patient ID is not the folder name",
         sum(1 for r in in_folder if _differs_from_folder(r)),
         sum(1 for r in done_in_folder if _not_pseudonym(r))),
        ("ECG files", len(raw), len(done)),
        ("ECG IDs", len({r["original_ecg_id"] for r in raw if r["original_ecg_id"]}),
         len({r["ecg_code"] for r in done if r["original_ecg_id"]})),
    ]


def format_stats(name: str, stats: list[tuple[str, int, int]]) -> str:
    width = max(len(s[0]) for s in stats)
    lines = [f"{name}", f"  {'':{width}}  {'raw':>10}  {'anonymized':>10}"]
    for what, before, after in stats:
        mark = "" if before == after else "  differs"
        lines.append(f"  {what:{width}}  {before:>10}  {after:>10}{mark}")
    return "\n".join(lines)

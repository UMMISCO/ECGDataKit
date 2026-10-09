"""CSV catalog of a dataset.

One row per raw file: original and pseudonymized values, paths, hashes,
status and dates. The catalog is the re-identification key of the dataset.
"""

from __future__ import annotations

import csv
import os
import threading
from pathlib import Path

COLUMNS = [
    "status",
    "risk",
    "risk_reason",
    "raw_path",
    "anonymized_path",
    "patient_folder",
    "format",
    "patient_code",
    "ecg_code",
    "original_patient_id",
    "original_last_name",
    "original_first_name",
    "original_ecg_id",
    "anonymized_ecg_id",
    "original_file_name",
    "anonymized_file_name",
    "folder_anonymized",
    "companions",
    "raw_size",
    "raw_mtime_ns",
    "raw_sha256",
    "anonymized_sha256",
    "detected_at",
    "anonymized_at",
    "last_change_at",
    "message",
    "tool_version",
]

ACTIVE = {"anonymized", "copied", "changed"}
"""Statuses of rows whose anonymized file exists."""

SEPARATOR = " | "


class Catalog:
    """Rows keyed by raw path (relative to the dataset folder).

    Writes go to a temporary file renamed over the catalog, so the CSV is
    never left half written.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.rows: dict[str, dict[str, str]] = {}
        self.lock = threading.RLock()
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.rows[row["raw_path"]] = {c: row.get(c, "") or "" for c in COLUMNS}

    def get(self, raw_path: str) -> dict[str, str] | None:
        with self.lock:
            return self.rows.get(raw_path)

    def put(self, row: dict[str, str]) -> None:
        with self.lock:
            self.rows[row["raw_path"]] = {c: str(row.get(c, "") or "") for c in COLUMNS}

    def active(self) -> list[dict[str, str]]:
        with self.lock:
            return [r for r in self.rows.values() if r["status"] in ACTIVE]

    def used_codes(self) -> set[str]:
        with self.lock:
            return {r[c] for r in self.rows.values() for c in ("patient_code", "ecg_code") if r[c]}

    def patient_codes(self) -> dict[str, str]:
        """Patient code by patient folder (every row ever written keeps its code)."""
        with self.lock:
            return {r["patient_folder"]: r["patient_code"] for r in self.rows.values()
                    if r["patient_folder"] and r["patient_code"]}

    def save(self) -> None:
        with self.lock:
            rows = sorted(self.rows.values(), key=lambda r: r["raw_path"])
            tmp = self.path.with_name(self.path.name + ".tmp")
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)

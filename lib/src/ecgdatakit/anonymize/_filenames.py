"""Anonymized file names.

Each anonymized file gets a new name made of its pseudonyms: nothing of
the raw name is kept, since a name can hold identity in any form.
"""

from __future__ import annotations


def split_name(name: str) -> tuple[str, str]:
    """Stem and extension (``.hea``, ``.xml``...)."""
    if "." not in name.lstrip("."):
        return name, ""
    stem, _, ext = name.rpartition(".")
    return stem, "." + ext


def anonymized_file_name(name: str, patient_code: str, ecg_code: str,
                         wfdb: bool = False) -> str:
    """Name of the anonymized file: the patient and ECG pseudonyms and the
    raw extension, ``<patient>_<ecg>.<ext>``. Nothing of the raw name is
    kept. WFDB record names only allow letters, digits and underscores, so
    the ECG pseudonym is written there without its hyphens."""
    _, ext = split_name(name)
    return f"{patient_code}_{ecg_code.replace('-', '') if wfdb else ecg_code}{ext}"

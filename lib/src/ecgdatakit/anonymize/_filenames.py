"""Anonymized file names.

Parts of the name equal to the file's name, patient ID or ECG ID fields
become the pseudonym; dates, visits and every other part are kept. Parts
that may still identify the patient are kept and reported as risks.
"""

from __future__ import annotations

import re

from ecgdatakit.anonymize._identity import Identity, Replacer, name_tokens, norm

_WORD_RE = re.compile(r"[^\W_]+")
_HASH_RE = re.compile(r"(?=.*\d)(?=.*[A-Za-z])[A-Za-z0-9]{16,}")
_VISIT_RE = re.compile(r"[vV]\d{1,3}")


def split_name(name: str, compound_suffix: bool = False) -> tuple[str, str]:
    """Stem and extension (``.hea``, ``.xml``...)."""
    if "." not in name.lstrip("."):
        return name, ""
    stem, _, ext = name.rpartition(".")
    return stem, "." + ext


def new_file_name(name: str, identity: Identity, patient_code: str, ecg_code: str,
                  folder_names: list[str], extra_ids: list[str] | None = None,
                  ) -> tuple[str, list[str]]:
    """Return the anonymized file name and the risks found in it.

    Parts equal to the file's name, patient ID or ECG ID fields (ignoring
    case, accents and surrounding spaces) become the pseudonym. When the file
    has no name field, the words of the patient folder name are used instead.
    Dates, visits and every other part are kept. Parts that may still be a
    name or an identifier (initials, long random strings) are kept and
    reported as risks.
    """
    stem, ext = split_name(name)
    new_stem, risks = new_part_name(stem, identity, patient_code, ecg_code, folder_names,
                                    extra_ids)
    return new_stem + ext, risks


def new_part_name(stem: str, identity: Identity, patient_code: str, ecg_code: str,
                  folder_names: list[str], extra_ids: list[str] | None = None,
                  ) -> tuple[str, list[str]]:
    """Anonymized form of one name without extension (a file stem or a
    folder name), with the risks found in it. See :func:`new_file_name`."""
    folder_tokens = name_tokens(*folder_names, min_len=2)
    extra = folder_tokens if not (identity.last_names or identity.first_names) else []
    replacer = Replacer(identity, patient_code, ecg_code, extra_names=extra,
                        min_name=2, min_id=2, extra_ids=extra_ids)
    new_stem = replacer.replace(stem)
    risks: list[str] = []
    if extra and new_stem != stem:
        risks.append("name parts matched the patient folder name, not a field of the file")

    names = {norm(t) for t in name_tokens(*identity.last_names, *identity.first_names,
                                          *folder_tokens, min_len=2)}
    codes = {patient_code, ecg_code}
    for word in _WORD_RE.findall(new_stem):
        if word in codes or word.isdigit() or _VISIT_RE.fullmatch(word):
            continue
        low = norm(word)
        if low in names:
            risks.append(f"name part {word!r} matches a name word")
        elif word.isalpha() and len(word) >= 2 and any(
                n.startswith(low) or low.startswith(n) for n in names if len(n) >= 2):
            risks.append(f"name part {word!r} may be initials or part of a name")
        elif _HASH_RE.fullmatch(word):
            risks.append(f"name part {word!r} may be an identifier")
    if not new_stem.strip(" ._-^"):
        new_stem = patient_code
    return new_stem, risks

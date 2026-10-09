"""Base class of the format handlers and shared byte helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ecgdatakit.anonymize._identity import Identity, Replacer

# Values that are sample data, never free text: only digits, signs,
# separators, or a long run without spaces (Base64, hex)
_NUMERIC_RE = re.compile(r"[\s\d.,;:+\-eE]*")
_BLOB_RE = re.compile(r"\S{200,}")


def is_data(value: str) -> bool:
    """True for sample data that free-text replacement must not touch."""
    return bool(_NUMERIC_RE.fullmatch(value)) or bool(_BLOB_RE.search(value))


@dataclass
class Codes:
    patient: str
    ecg: str


@dataclass(frozen=True)
class Spliced:
    """Output made of new header bytes followed by the source from *offset*."""

    head: bytes
    source: Path
    offset: int


@dataclass
class Rewrite:
    """Files written for one source file and what was changed."""

    outputs: dict[Path, bytes | Spliced] = field(default_factory=dict)
    """Anonymized content by output path (bytes, or a header spliced onto
    the rest of the source for large files)."""
    ecg_written: list[str] = field(default_factory=list)
    """ECG ID values as written in the output (UUID, UID or code)."""
    notes: list[str] = field(default_factory=list)
    """Risks found while rewriting (kept as is, reported in the catalog)."""


class Handler:
    """Reads identity values and writes an anonymized copy of one format."""

    parser: str = ""
    """Name of the ecgdatakit parser class whose files this handler takes."""

    def identity(self, path: Path) -> Identity:
        raise NotImplementedError

    def companions(self, path: Path) -> list[Path]:
        """Other files that belong to *path* (WFDB signal files)."""
        return []

    def rewrite(self, path: Path, out_path: Path, identity: Identity, codes: Codes,
                replacer: Replacer) -> Rewrite:
        raise NotImplementedError

    def free_text(self, data: bytes) -> list[str]:
        """Field values of *data* that may hold identity text (for the check
        of identity text left in the anonymized file)."""
        raise NotImplementedError


def fixed_width(text: str, size: int, encoding: str = "latin-1", pad: bytes = b"\x00") -> bytes:
    """*text* encoded and padded (or cut) to *size* bytes."""
    raw = text.encode(encoding, errors="replace")[:size]
    return raw + pad * (size - len(raw))


def replace_same_length(region: bytes, replacer: Replacer, encoding: str = "latin-1") -> bytes:
    """Free-text replacement that keeps every byte position.

    Used inside fixed-size binary fields: each match is replaced by the code,
    cut or padded with spaces to the length of the match.
    """
    if not replacer or not region:
        return region
    text = region.decode(encoding, errors="surrogateescape")
    regex = replacer._regex  # noqa: SLF001 - same module family

    def same_length(match: re.Match) -> str:
        code = replacer._code_for(match)  # noqa: SLF001
        size = len(match.group(0).encode(encoding, errors="surrogateescape"))
        return code[:size].ljust(size)

    new = regex.sub(same_length, text)
    out = new.encode(encoding, errors="surrogateescape")
    return out if len(out) == len(region) else region

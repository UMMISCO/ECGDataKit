"""Identity values of a file and their replacement in free text.

Matching ignores case, accents and repeated spaces, and a match must not
touch a letter or digit on either side. Name words shorter than 3
characters and IDs shorter than 4 are not searched in free text, to avoid
replacing unrelated text.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Free text: name parts shorter than this are not searched (too many false hits)
MIN_NAME_TOKEN = 3
# Free text: patient IDs and ECG IDs shorter than this are not searched
MIN_ID_TOKEN = 4

_SPLIT_RE = re.compile(r"[\s,;^_/\\.\-]+")


def norm(text: str) -> str:
    """Lowercase, strip and drop accents, for matching only."""
    text = unicodedata.normalize("NFKD", " ".join(text.split()).casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


def name_tokens(*names: str, min_len: int = MIN_NAME_TOKEN) -> list[str]:
    """Words of the given names, longest first, without duplicates."""
    tokens = {t for n in names if n for t in _SPLIT_RE.split(n.strip()) if len(t) >= min_len}
    return sorted(tokens, key=lambda t: (-len(t), t))


@dataclass
class Identity:
    """Identity values stored in one file.

    Every value is kept as written in the file, so the CSV catalog holds the
    exact original text.
    """

    patient_ids: list[str] = field(default_factory=list)
    last_names: list[str] = field(default_factory=list)
    first_names: list[str] = field(default_factory=list)
    ecg_ids: list[str] = field(default_factory=list)

    def add(self, kind: str, value: str | None) -> None:
        value = (value or "").strip()
        values = getattr(self, kind)
        if value and value not in values:
            values.append(value)

    def merge(self, other: Identity) -> None:
        for kind in ("patient_ids", "last_names", "first_names", "ecg_ids"):
            for value in getattr(other, kind):
                self.add(kind, value)

    @property
    def has_patient(self) -> bool:
        return bool(self.patient_ids or self.last_names or self.first_names)

    @property
    def is_empty(self) -> bool:
        return not (self.has_patient or self.ecg_ids)

    def full_names(self) -> list[str]:
        """Whole-name strings in the usual orders (``First Last``, ``Last First``...)."""
        names = set(self.last_names) | set(self.first_names)
        for last in self.last_names:
            for first in self.first_names:
                for sep in (" ", ", ", ",", "^", "_", " ^"):
                    names.add(f"{last}{sep}{first}")
                names.add(f"{first} {last}")
        return sorted((n for n in names if n), key=lambda n: (-len(n), n))


class Replacer:
    """Replace identity values inside free text with pseudonyms.

    Whole values are replaced first (longest first), then single name words.
    A match must not touch a letter or digit on either side, and case and
    accents are ignored.
    """

    def __init__(self, identity: Identity, patient_code: str, ecg_code: str,
                 extra_names: list[str] | None = None, min_name: int = MIN_NAME_TOKEN,
                 min_id: int = MIN_ID_TOKEN) -> None:
        pairs: list[tuple[str, str]] = []
        for value in identity.ecg_ids:
            if len(value) >= min_id:
                pairs.append((value, ecg_code))
        for value in identity.patient_ids:
            if len(value) >= min_id:
                pairs.append((value, patient_code))
        for value in identity.full_names():
            if len(value) >= min_name:
                pairs.append((value, patient_code))
        names = identity.last_names + identity.first_names + list(extra_names or [])
        for token in name_tokens(*names, min_len=min_name):
            pairs.append((token, patient_code))
        # Whole values compared as is, also in fields that look like numbers
        self.exact: dict[str, str] = {}
        for value in identity.patient_ids:
            self.exact[norm(value)] = patient_code
        for value in identity.ecg_ids:
            self.exact[norm(value)] = ecg_code
        seen: set[str] = set()
        self.pairs: list[tuple[str, str]] = []
        for value, code in sorted(pairs, key=lambda p: -len(p[0])):
            key = norm(value)
            if key and key not in seen:
                seen.add(key)
                self.pairs.append((value, code))
        self._regex = None
        if self.pairs:
            self._codes = {norm(v): c for v, c in self.pairs}
            alternatives = "|".join(_loose(v) for v, _ in self.pairs)
            self._regex = re.compile(
                rf"(?<![^\W_])(?:{alternatives})(?![^\W_])", re.IGNORECASE)

    def __bool__(self) -> bool:
        return self._regex is not None

    def replace(self, text: str) -> str:
        if not text or self._regex is None:
            return text
        return self._regex.sub(self._code_for, text)

    def replace_value(self, text: str, data_like: bool) -> str:
        """Replace in a field value.

        Free text gets :meth:`replace`. A value that looks like sample data
        (numbers, Base64) is replaced only when the whole value equals a
        patient ID or ECG ID, and the ID is not a plain number shorter than
        6 digits (which could be a measurement).
        """
        if not data_like:
            return self.replace(text)
        code = self.exact.get(norm(text))
        if code is None:
            return text
        stripped = text.strip()
        if stripped.isdigit() and len(stripped) < 6:
            return text
        return text.replace(stripped, code)

    def count(self, text: str) -> int:
        if not text or self._regex is None:
            return 0
        return sum(1 for _ in self._regex.finditer(text))

    def _code_for(self, match: re.Match) -> str:
        return self._codes.get(norm(match.group(0)), self.pairs[0][1])


def _loose(value: str) -> str:
    """Regex for *value* where accented letters also match their plain form."""
    out = []
    for ch in value:
        base = unicodedata.normalize("NFKD", ch)[0]
        if base != ch and base.isalpha():
            out.append(f"[{re.escape(ch)}{re.escape(base)}]")
        elif ch.isspace():
            out.append(r"\s+")
        else:
            out.append(re.escape(ch))
    return "".join(out)

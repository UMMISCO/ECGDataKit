"""Lead label and text helpers shared by all parsers."""

from __future__ import annotations

STANDARD_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
"""The 12 standard leads in conventional order."""

_EXTRA_LEADS = (
    "V7", "V8", "V9", "V3R", "V4R", "V5R", "V6R", "V7R", "V8R", "V9R",
    "X", "Y", "Z", "-aVR",
)

_CANONICAL = {name.lower(): name for name in STANDARD_LEADS + _EXTRA_LEADS}

_PREFIXES = ("ecg lead ", "ecg ", "lead ")


def normalize_lead_label(label: str) -> str:
    """Return the standard spelling of a lead label (``"AVR"`` -> ``"aVR"``).

    Prefixes such as ``"ECG "`` or ``"Lead "`` are removed. Labels that are
    not a known lead name are returned stripped but otherwise unchanged.
    """
    text = label.strip()
    key = text.lower()
    for prefix in _PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix):].strip()
            break
    return _CANONICAL.get(key, text)


def unique_labels(labels: list[str]) -> list[str]:
    """Suffix repeated labels so every label is unique (``II``, ``II_2``, ...).

    A suffix never reuses a label present in the input, so
    ``["II", "II_2", "II"]`` gives ``["II", "II_2", "II_3"]``.
    """
    originals = set(labels)
    used: set[str] = set()
    counts: dict[str, int] = {}
    result: list[str] = []
    for label in labels:
        name = label
        if name in used:
            n = counts.get(label, 1)
            while name in used or name in originals:
                n += 1
                name = f"{label}_{n}"
            counts[label] = n
        used.add(name)
        result.append(name)
    return result


def decode_text(raw: bytes, encoding: str | None = None) -> str:
    """Decode bytes without losing characters.

    Tries *encoding* (when given) then UTF-8, and falls back to Latin-1,
    which maps every byte to one character.
    """
    for enc in (encoding, "utf-8"):
        if not enc:
            continue
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin-1")

"""Parser framework with auto-discovery of format-specific parsers."""

from __future__ import annotations

import dataclasses
import functools
import importlib
import pkgutil
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ecgdatakit.exceptions import (
    CorruptedFileError,
    ECGDataKitError,
    UnsupportedFormatError,
)
from ecgdatakit.models import ECGRecord, Lead, _TO_UV, _normalize_unit
from ecgdatakit.parsing.helpers.record import fill_signal_summary

_SNIFF_SIZE = 4096


class Parser(ABC):
    """Base class for all ECG format parsers."""

    FORMAT_NAME: str = ""
    FORMAT_DESCRIPTION: str = ""
    FILE_EXTENSIONS: list[str] = []
    PRIORITY: int = 50
    """Detection order in :class:`FileParser`, lower runs first. Formats with
    a reliable magic number use a low value, loose sniffers a high one."""

    @staticmethod
    @abstractmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        """Check if this parser handles the given file.

        Parameters
        ----------
        file_path : Path
            Path to the ECG file.
        header : bytes
            First 4096 bytes of the file for format sniffing.
        """
        ...

    @abstractmethod
    def parse(self, file_path: Path) -> ECGRecord:
        """Parse the file and return a structured ECGRecord."""
        ...


@functools.cache
def _discover() -> tuple[type[Parser], ...]:
    """Every Parser subclass of ``ecgdatakit.parsing.parsers``, in
    detection order."""
    package = importlib.import_module("ecgdatakit.parsing.parsers")
    found: list[type[Parser]] = []
    for _, name, _ in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"ecgdatakit.parsing.parsers.{name}")
        for attr in vars(module).values():
            if (
                isinstance(attr, type)
                and issubclass(attr, Parser)
                and attr is not Parser
            ):
                found.append(attr)
    return tuple(sorted(found, key=lambda cls: (cls.PRIORITY, cls.__name__)))


class FileParser:
    """Auto-discovers parsers and dispatches files to the right one.

    No method needs an instance: ``FileParser.parse(path)`` and
    ``FileParser().parse(path)`` do the same.
    """

    @property
    def parsers(self) -> list[type[Parser]]:
        """List of discovered :class:`Parser` subclasses."""
        return list(_discover())

    @staticmethod
    def supported_formats() -> list[dict[str, str | list[str]]]:
        """Return a description of every supported ECG format.

        Can be called without instantiation::

            FileParser.supported_formats()

        Each entry contains:

        - ``name`` – short format name (e.g. ``"HL7 aECG"``)
        - ``description`` – one-line description
        - ``extensions`` – list of typical file extensions
        """
        package = importlib.import_module("ecgdatakit.parsing.parsers")
        parsers: list[type[Parser]] = []
        for _, name, _ in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"ecgdatakit.parsing.parsers.{name}")
            for attr in vars(module).values():
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Parser)
                    and attr is not Parser
                ):
                    parsers.append(attr)
        return [
            {
                "name": p.FORMAT_NAME or p.__name__,
                "description": p.FORMAT_DESCRIPTION or (p.__doc__ or "").strip(),
                "extensions": list(p.FILE_EXTENSIONS),
            }
            for p in parsers
        ]

    @staticmethod
    def detect_format(file_path: str | Path) -> dict | None:
        """Detect the format of an ECG file without parsing it.

        Returns the format and the parser :meth:`parse` would use, or
        ``None`` when no parser recognises the file. The dict holds
        ``name``, ``description`` and ``extensions`` (as in
        :meth:`supported_formats`), ``parser`` (the class name, e.g.
        ``"SCPECGParser"``) and ``parser_class`` (the class itself)::

            fmt = FileParser.detect_format("ecg.scp")
            record = fmt["parser_class"]().parse("ecg.scp")

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        parser_cls = FileParser._find_parser(path)
        if parser_cls is None:
            return None
        return {
            "name": parser_cls.FORMAT_NAME or parser_cls.__name__,
            "description": parser_cls.FORMAT_DESCRIPTION or (parser_cls.__doc__ or "").strip(),
            "extensions": list(parser_cls.FILE_EXTENSIONS),
            "parser": parser_cls.__name__,
            "parser_class": parser_cls,
        }

    @staticmethod
    def _find_parser(path: Path) -> type[Parser] | None:
        """First parser, in detection order, that recognises *path*."""
        header = b""
        if path.is_file():
            with open(path, "rb") as f:
                header = f.read(_SNIFF_SIZE)
        return next((p for p in _discover() if p.can_parse(path, header)), None)

    @staticmethod
    def parse(
        file_path: str | Path,
        auto_scale: bool = True,
        units: str = "mV",
    ) -> ECGRecord:
        """Parse an ECG file, auto-detecting the format.

        Parameters
        ----------
        file_path : str | Path
            Path to the ECG file.
        auto_scale : bool
            When ``True`` (default), leads with scaling metadata are
            automatically converted to physical units (see *units*).
            Leads without sufficient metadata are left as raw ADC
            values and a warning is emitted.  Set to ``False`` to
            always receive raw ADC samples.
        units : str
            Target voltage unit when *auto_scale* is ``True``.
            Accepted values: ``"uV"`` (microvolts), ``"mV"``
            (millivolts, default), ``"V"`` (volts), in any case.
            Ignored when *auto_scale* is ``False``.

        Raises
        ------
        UnsupportedFormatError
            If no parser can handle the file (also a ``ValueError``).
        ValueError
            If *units* is not recognised.
        CorruptedFileError
            If the file is recognised but cannot be decoded, or holds no
            samples.
        ImportError
            If the format needs an optional dependency that is missing.
        """
        # Validate units early
        target = _normalize_unit(units)
        if target not in ("uV", "mV", "V"):
            raise ValueError(
                f"Unknown unit {units!r}. "
                "Accepted values: 'uV', 'mV', 'V'."
            )

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        parser_cls = FileParser._find_parser(path)
        if parser_cls is None:
            raise UnsupportedFormatError(f"No parser found for: {path.name}")

        # Parser warnings are re-emitted here so they point at the caller,
        # also when the parser fails
        caught: list[warnings.WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                record = FileParser._run_parser(parser_cls, path)
                if record.leads and all(len(lead.samples) == 0 for lead in record.leads):
                    raise CorruptedFileError(
                        f"{parser_cls.FORMAT_NAME or parser_cls.__name__}: "
                        f"{path.name} holds no samples"
                    )
                fill_signal_summary(record)
                if auto_scale:
                    record = FileParser._auto_scale(record, target)
        finally:
            for w in caught:
                warnings.warn(w.message, w.category, stacklevel=2)

        if not auto_scale and any(lead.is_raw for lead in record.leads):
            warnings.warn(
                "auto_scale=False: leads contain raw ADC samples. "
                "Amplitudes are unitless and not in physical units (mV).",
                stacklevel=2,
            )
        record.raw_metadata.setdefault("filepath", str(path))
        return record

    @staticmethod
    def _run_parser(parser_cls: type[Parser], path: Path) -> ECGRecord:
        """Run *parser_cls* and report decoding failures as CorruptedFileError."""
        try:
            record = parser_cls().parse(path)
        except (ECGDataKitError, OSError, ImportError):
            # ImportError: a missing optional dependency, not a corrupt file
            raise
        except Exception as e:
            raise CorruptedFileError(
                f"{parser_cls.FORMAT_NAME or parser_cls.__name__}: "
                f"cannot decode {path.name}: {type(e).__name__}: {e}"
            ) from e
        record.file_format.name = parser_cls.FORMAT_NAME or parser_cls.__name__
        record.file_format.parser = parser_cls.__name__
        return record

    @staticmethod
    def _auto_scale(record: ECGRecord, target: str = "mV") -> ECGRecord:
        """Convert leads to physical units where scaling metadata is available.

        The record comes straight from a parser and is not shared, so its
        float64 sample arrays are scaled in place: no second copy of a long
        recording is made. Other arrays (other dtypes, read-only, or shared
        between leads) are copied.

        Parameters
        ----------
        record : ECGRecord
            Parsed record with raw or partially-scaled leads.
        target : str
            Canonical target unit (``"uV"``, ``"mV"``, or ``"V"``).
        """
        raw_labels: list[str] = []
        other_units: list[str] = []
        converted: list[str] = []
        def root(samples: np.ndarray) -> int:
            while isinstance(samples.base, np.ndarray):
                samples = samples.base
            return id(samples)

        # Buffers used by more than one lead must not be modified in place
        users: dict[int, int] = {}
        for lead in record.leads + record.median_beats + record.leads_enhanced:
            key = root(np.asarray(lead.samples))
            users[key] = users.get(key, 0) + 1

        def owned(samples: np.ndarray) -> bool:
            # True when the array may be modified in place
            return (
                isinstance(samples, np.ndarray)
                and samples.dtype == np.float64
                and samples.flags.writeable
                and users.get(root(samples)) == 1
            )

        def apply(samples: np.ndarray, scale: float, shift: float) -> np.ndarray:
            if owned(samples):
                samples *= scale
                if shift:
                    samples += shift
                return samples
            return samples * scale + shift

        def scale(lead: Lead, report: bool) -> Lead:
            if lead.is_raw:
                # A raw lead needs a known voltage unit and a non-zero resolution
                if not lead.resolution_unit or lead.resolution == 0.0:
                    if report:
                        raw_labels.append(lead.label)
                    return lead
                unit = _normalize_unit(lead.resolution_unit)
                if unit is None:
                    if report:
                        other_units.append(f"{lead.label} ({lead.resolution_unit})")
                    return lead.to_physical()
                factor = _TO_UV[unit] / _TO_UV[target]
                converted.append(lead.label)
                return dataclasses.replace(
                    lead,
                    samples=apply(lead.samples, lead.resolution * factor, lead.offset * factor),
                    is_raw=False,
                    units=target,
                )
            unit = _normalize_unit(lead.units)
            if unit is None:
                if report:
                    other_units.append(f"{lead.label} ({lead.units or 'no unit'})")
                return lead
            if unit == target:
                return dataclasses.replace(lead, units=target)
            factor = _TO_UV[unit] / _TO_UV[target]
            return dataclasses.replace(
                lead, samples=apply(lead.samples, factor, 0.0), units=target,
            )

        record.leads = [scale(lead, True) for lead in record.leads]
        record.median_beats = [scale(beat, False) for beat in record.median_beats]
        record.leads_enhanced = [scale(lead, False) for lead in record.leads_enhanced]

        if raw_labels:
            warnings.warn(
                f"Leads {raw_labels} contain raw ADC samples, no scaling "
                "metadata available. Pass auto_scale=False to get raw values.",
                stacklevel=3,
            )
        if other_units:
            warnings.warn(
                f"Leads {other_units} are not in a voltage unit and were not "
                f"converted to {target}.",
                stacklevel=3,
            )
        # A scale the file does not state, set by the parser (experimental)
        scale_warning = record.raw_metadata.get("scale_warning")
        if converted and scale_warning:
            warnings.warn(scale_warning, UserWarning, stacklevel=3)
        return record

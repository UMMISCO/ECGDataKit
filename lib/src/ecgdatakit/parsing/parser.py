"""Parser framework with auto-discovery of format-specific parsers."""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

from ecgdatakit.models import ECGRecord, _UNIT_ALIASES


class Parser(ABC):
    """Base class for all ECG format parsers."""

    FORMAT_NAME: str = ""
    FORMAT_DESCRIPTION: str = ""
    FILE_EXTENSIONS: list[str] = []

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


class FileParser:
    """Auto-discovers parsers and dispatches files to the right one."""

    def __init__(self) -> None:
        self._parsers: list[type[Parser]] = []
        self._discover_parsers()

    def _discover_parsers(self) -> None:
        """Find all Parser subclasses in ecgdatakit.parsers package."""
        package = importlib.import_module("ecgdatakit.parsing.parsers")
        for _, name, _ in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"ecgdatakit.parsing.parsers.{name}")
            for attr in vars(module).values():
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Parser)
                    and attr is not Parser
                ):
                    self._parsers.append(attr)

    @property
    def parsers(self) -> list[type[Parser]]:
        """List of discovered :class:`Parser` subclasses."""
        return list(self._parsers)

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

    def parse(
        self, file_path: str | Path, auto_scale: bool = True,
    ) -> ECGRecord:
        """Parse an ECG file, auto-detecting the format.

        Parameters
        ----------
        file_path : str | Path
            Path to the ECG file.
        auto_scale : bool
            When ``True`` (default), leads with scaling metadata are
            automatically converted to millivolts (``mV``).  Leads
            without sufficient metadata are left as raw ADC values and
            a warning is emitted.  Set to ``False`` to always receive
            raw ADC samples.

        Raises
        ------
        ValueError
            If no parser can handle the file.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        header = path.read_bytes()[:4096]
        for parser_cls in self._parsers:
            if parser_cls.can_parse(path, header):
                record = parser_cls().parse(path)
                if auto_scale:
                    return self._auto_scale(record)
                return record
        raise ValueError(f"No parser found for: {path.name}")

    @staticmethod
    def _auto_scale(record: ECGRecord) -> ECGRecord:
        """Convert leads to mV where scaling metadata is available."""
        import dataclasses

        new_leads = []
        raw_labels: list[str] = []
        for lead in record.leads:
            if lead.resolution == 1.0 and lead.offset == 0.0 and not lead.units:
                raw_labels.append(lead.label)
                new_leads.append(lead)
                continue
            physical = lead.to_physical()
            norm = _UNIT_ALIASES.get(physical.units)
            if norm and norm != "mV":
                physical = physical.convert_units("mV")
            new_leads.append(physical)

        new_beats = []
        for beat in record.median_beats:
            if beat.resolution == 1.0 and beat.offset == 0.0 and not beat.units:
                new_beats.append(beat)
                continue
            physical = beat.to_physical()
            norm = _UNIT_ALIASES.get(physical.units)
            if norm and norm != "mV":
                physical = physical.convert_units("mV")
            new_beats.append(physical)

        if raw_labels:
            warnings.warn(
                f"Leads {raw_labels} contain raw ADC samples — no scaling "
                "metadata available. Pass auto_scale=False to get raw values.",
                stacklevel=3,
            )

        return dataclasses.replace(
            record, leads=new_leads, median_beats=new_beats,
        )

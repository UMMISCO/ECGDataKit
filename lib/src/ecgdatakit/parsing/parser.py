"""Parser framework with auto-discovery of format-specific parsers."""

from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from pathlib import Path

from ecgdatakit.models import ECGRecord


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
        return list(self._parsers)

    def supported_formats(self) -> list[dict[str, str | list[str]]]:
        """Return a description of every supported ECG format.

        Each entry contains:

        - ``name`` – short format name (e.g. ``"HL7 aECG"``)
        - ``description`` – one-line description
        - ``extensions`` – list of typical file extensions
        """
        return [
            {
                "name": p.FORMAT_NAME or p.__name__,
                "description": p.FORMAT_DESCRIPTION or (p.__doc__ or "").strip(),
                "extensions": list(p.FILE_EXTENSIONS),
            }
            for p in self._parsers
        ]

    def parse(self, file_path: str | Path) -> ECGRecord:
        """Parse an ECG file, auto-detecting the format.

        Parameters
        ----------
        file_path : str | Path
            Path to the ECG file.

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
                return parser_cls().parse(path)
        raise ValueError(f"No parser found for: {path.name}")

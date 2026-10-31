"""Base class of the exporters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ecgdatakit.models import ECGRecord


class Exporter(ABC):
    """Write an :class:`~ecgdatakit.models.ECGRecord` to a file format."""

    FORMAT_NAME: str = ""
    """Name of the format written."""
    FILE_EXTENSION: str = ""
    """Usual extension of the files written."""

    @abstractmethod
    def to_bytes(self, record: ECGRecord) -> bytes:
        """The file content for *record*."""

    def export(self, record: ECGRecord, path: str | Path) -> Path:
        """Write *record* to *path* and return the path.

        The file is written only once its whole content is built, so a
        record that cannot be exported leaves no partial file.
        """
        content = self.to_bytes(record)
        path = Path(path)
        path.write_bytes(content)
        return path

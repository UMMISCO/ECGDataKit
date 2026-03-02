"""ECGDataKit - Multi-format ECG file parsing and processing library."""

__version__ = "0.0.6"

from ecgdatakit.models import (
    DeviceInfo,
    ECGRecord,
    FilterSettings,
    GlobalMeasurements,
    Interpretation,
    Lead,
    PatientInfo,
    RecordingInfo,
)
from ecgdatakit.parsing.parser import FileParser, Parser
from ecgdatakit.exceptions import (
    ECGDataKitError,
    UnsupportedFormatError,
    CorruptedFileError,
    MissingElementError,
    ChecksumError,
)
from ecgdatakit.parsing.batch import parse_batch

__all__ = [
    "DeviceInfo",
    "ECGRecord",
    "FilterSettings",
    "GlobalMeasurements",
    "Interpretation",
    "Lead",
    "PatientInfo",
    "RecordingInfo",
    "FileParser",
    "Parser",
    "ECGDataKitError",
    "UnsupportedFormatError",
    "CorruptedFileError",
    "MissingElementError",
    "ChecksumError",
    "parse_batch",
]

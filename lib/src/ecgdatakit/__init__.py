"""ECGDataKit - Multi-format ECG file parsing and processing library."""

__version__ = "0.0.8"

from ecgdatakit.models import (
    DeviceInfo,
    ECGRecord,
    FilterSettings,
    GlobalMeasurements,
    Interpretation,
    Lead,
    LeadLike,
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
    RawSamplesError,
)
from ecgdatakit.parsing.batch import parse_batch

__all__ = [
    "DeviceInfo",
    "ECGRecord",
    "FilterSettings",
    "GlobalMeasurements",
    "Interpretation",
    "Lead",
    "LeadLike",
    "PatientInfo",
    "RecordingInfo",
    "FileParser",
    "Parser",
    "ECGDataKitError",
    "UnsupportedFormatError",
    "CorruptedFileError",
    "MissingElementError",
    "ChecksumError",
    "RawSamplesError",
    "parse_batch",
]

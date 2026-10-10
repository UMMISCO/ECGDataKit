"""ECGDataKit - Multi-format ECG file parsing and processing library."""

__version__ = "2.0.0"

from ecgdatakit.models import (
    AcquisitionSetup,
    DeviceInfo,
    ECGRecord,
    FileFormatInfo,
    FilterSettings,
    GlobalMeasurements,
    Interpretation,
    Lead,
    LeadLike,
    PatientInfo,
    RecordingInfo,
    SignalCharacteristics,
    derive_is_raw,
)
from ecgdatakit.parsing.parser import FileParser, Parser
from ecgdatakit.exceptions import (
    ECGDataKitError,
    UnsupportedFormatError,
    CorruptedFileError,
    MissingElementError,
    ChecksumError,
    ChecksumWarning,
    RawSamplesError,
)
from ecgdatakit.parsing.batch import BatchParseWarning, parse_batch

__all__ = [
    "AcquisitionSetup",
    "DeviceInfo",
    "ECGRecord",
    "FileFormatInfo",
    "FilterSettings",
    "GlobalMeasurements",
    "Interpretation",
    "Lead",
    "LeadLike",
    "PatientInfo",
    "RecordingInfo",
    "SignalCharacteristics",
    "derive_is_raw",
    "FileParser",
    "Parser",
    "ECGDataKitError",
    "UnsupportedFormatError",
    "CorruptedFileError",
    "MissingElementError",
    "ChecksumError",
    "ChecksumWarning",
    "RawSamplesError",
    "BatchParseWarning",
    "parse_batch",
]

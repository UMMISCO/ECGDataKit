"""ECGDataKit - Multi-format ECG file parsing and processing library."""

__version__ = "2.0.4"

from ecgdatakit.models import (
    AcquisitionSetup,
    BeatAnnotations,
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
from ecgdatakit.parsing.parsers.ishne_holter import read_ishne_annotations

__all__ = [
    "AcquisitionSetup",
    "BeatAnnotations",
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
    "read_ishne_annotations",
]

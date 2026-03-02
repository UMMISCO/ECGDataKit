"""Custom exceptions for ecgdatakit."""


class ECGDataKitError(Exception):
    """Base exception for all ecgdatakit errors."""


class UnsupportedFormatError(ECGDataKitError):
    """File format is not recognized or not supported."""


class CorruptedFileError(ECGDataKitError):
    """File appears corrupted or truncated."""


class MissingElementError(ECGDataKitError):
    """Expected XML element or binary field is missing."""


class ChecksumError(ECGDataKitError):
    """File checksum validation failed."""

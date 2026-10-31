"""Custom exceptions for ecgdatakit."""


class ECGDataKitError(Exception):
    """Base exception for all ecgdatakit errors."""


class UnsupportedFormatError(ECGDataKitError, ValueError):
    """File format is not recognized or not supported.

    Also a :class:`ValueError`, which :meth:`FileParser.parse` raised for
    unknown files before this exception existed.
    """


class CorruptedFileError(ECGDataKitError):
    """File appears corrupted or truncated."""


class MissingElementError(ECGDataKitError):
    """Expected XML element or binary field is missing."""


class ChecksumError(ECGDataKitError):
    """File checksum validation failed."""


class RawSamplesError(ECGDataKitError):
    """Operation requires physical-unit samples but samples are still raw ADC.

    Call :meth:`Lead.to_physical` or :meth:`ECGRecord.to_physical` first.
    """


class ExportError(ECGDataKitError):
    """A record cannot be written to the target format, or the written file
    does not conform to the format's schema."""


class ChecksumWarning(UserWarning):
    """Stored checksum does not match the computed one.

    Silence it in batch processing with
    ``warnings.filterwarnings("ignore", category=ChecksumWarning)``.
    """

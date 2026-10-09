# Exceptions

All exceptions inherit from {class}`~ecgdatakit.exceptions.ECGDataKitError`.

Import: `from ecgdatakit import ECGDataKitError, UnsupportedFormatError, ...`

{class}`~ecgdatakit.exceptions.ECGDataKitError`
: Base exception for all ECGDataKit errors.

{class}`~ecgdatakit.exceptions.UnsupportedFormatError`
: File format not recognized by any parser, or a recognized format variant that is not supported. It also subclasses `ValueError`.

{class}`~ecgdatakit.exceptions.CorruptedFileError`
: File is truncated or structurally invalid.

{class}`~ecgdatakit.exceptions.MissingElementError`
: Required element or field is missing from the file.

{class}`~ecgdatakit.exceptions.ChecksumError`
: Checksum or CRC validation failed.

{class}`~ecgdatakit.exceptions.RawSamplesError`
: Operation requires physical-unit samples but the lead still contains raw ADC values. Call `to_physical()` first.

A missing optional dependency (for example pydicom for DICOM files) raises `ImportError` with the install command.

## Warnings

{class}`~ecgdatakit.exceptions.ChecksumWarning`
: Stored checksum does not match the computed one. The file is still parsed and `raw_metadata["checksum_valid"]` is set to `False`. Some devices never write a real checksum, so in batch processing you can silence it with `warnings.filterwarnings("ignore", category=ChecksumWarning)`.

## Example

```python
from ecgdatakit import FileParser, UnsupportedFormatError

try:
    record = FileParser().parse("unknown.bin")
except UnsupportedFormatError as e:
    print(f"Format not supported: {e}")
```

```python
from ecgdatakit.exceptions import RawSamplesError

try:
    lead.convert_units("mV")
except RawSamplesError:
    lead = lead.to_physical().convert_units("mV")
```

## Full API

```{eval-rst}
.. currentmodule:: ecgdatakit.exceptions

.. autoexception:: ECGDataKitError
.. autoexception:: UnsupportedFormatError
.. autoexception:: CorruptedFileError
.. autoexception:: MissingElementError
.. autoexception:: ChecksumError
.. autoexception:: RawSamplesError
.. autoexception:: ChecksumWarning
```

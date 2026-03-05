# Exceptions

All exceptions inherit from {class}`~ecgdatakit.exceptions.ECGDataKitError`.

Import: `from ecgdatakit import ECGDataKitError, UnsupportedFormatError, ...`

{class}`~ecgdatakit.exceptions.ECGDataKitError`
: Base exception for all ECGDataKit errors.

{class}`~ecgdatakit.exceptions.UnsupportedFormatError`
: File format not recognized by any parser.

{class}`~ecgdatakit.exceptions.CorruptedFileError`
: File is truncated or structurally invalid.

{class}`~ecgdatakit.exceptions.MissingElementError`
: Required element or field is missing from the file.

{class}`~ecgdatakit.exceptions.ChecksumError`
: Checksum or CRC validation failed.

{class}`~ecgdatakit.exceptions.RawSamplesError`
: Operation requires physical-unit samples but the lead still contains raw ADC values. Call `to_physical()` first.

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
```

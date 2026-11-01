# Parsing

Import: `from ecgdatakit import FileParser, parse_batch, BatchParseWarning, read_ishne_annotations`

| | |
|---|---|
| {class}`~ecgdatakit.parsing.parser.FileParser` | Auto-detect format and parse any supported ECG file |
| {class}`~ecgdatakit.parsing.parser.Parser` | Base class for all ECG format parsers |
| {func}`~ecgdatakit.parsing.batch.parse_batch` | Parse multiple ECG files in parallel |
| {class}`~ecgdatakit.parsing.batch.BatchParseWarning` | Warning for a file skipped by `parse_batch(on_error="warn")` |
| {func}`~ecgdatakit.parsing.parsers.ishne_holter.read_ishne_annotations` | Read the beat annotations of an ISHNE Holter (`.ann`) |

```{eval-rst}
.. currentmodule:: ecgdatakit.parsing.parser
```

## FileParser

```{eval-rst}
.. autoclass:: FileParser
   :members:
   :member-order: bysource
```

## Parser

```{eval-rst}
.. autoclass:: Parser
   :members:
   :member-order: bysource
```

## parse_batch

```{eval-rst}
.. currentmodule:: ecgdatakit.parsing.batch

.. autofunction:: parse_batch

.. autoclass:: BatchParseWarning
```

`parse_batch` keeps at most `2 * max_workers` records in memory and cancels pending files on `break` or error. A crashed worker counts as a failed file under `on_error`. Records cross processes by pickling, which is slow for long recordings: use `executor="serial"` for Holter files. Guard the call with `if __name__ == "__main__":`.

Unknown files raise `UnsupportedFormatError` (a `ValueError`). A missing optional dependency raises `ImportError`. A file with no samples raises `CorruptedFileError`.

## Examples

### Basic usage

```python
from ecgdatakit import FileParser

record = FileParser.parse("ecg_file.xml")
```

`FileParser` needs no instance. `FileParser.parse(path)` and `FileParser().parse(path)` do the same.

### Detecting the format

`detect_format` tells which format a file is in and which parser reads it, without parsing it. It returns `None` when no parser recognises the file.

```python
fmt = FileParser.detect_format("ecg_file.scp")
print(fmt["name"], fmt["parser"])          # SCP-ECG SCPECGParser

# Use the suggested parser directly, for example to pass format options
record = fmt["parser_class"]().parse("ecg_file.scp")
```

A parser called directly returns the values as stored, without the conversion to mV that `FileParser.parse` applies. Convert with `record.to_physical().convert_units("mV")`.

### Auto-scaling

`auto_scale` controls whether leads are automatically converted from raw ADC integers to millivolts.
When `True` (default), leads with scaling metadata (`resolution`, `offset`, `units`) are converted to **mV** automatically.
Leads without sufficient metadata are left as raw ADC values and a warning is emitted:

```
UserWarning: Leads ['Ch1', 'Ch2'] contain raw ADC samples, no scaling
metadata available. Pass auto_scale=False to get raw values.
```

With `auto_scale=False`, `lead.samples` are the values stored in the file, and
`resolution`, `offset` and `resolution_unit` describe the conversion
(`physical = samples * resolution + offset`).

Warnings raised by a parser point at the line that called `FileParser.parse`.
A file no parser recognises raises `UnsupportedFormatError`. Errors while
decoding a recognised file are raised as `CorruptedFileError`, and a missing
optional dependency (pydicom for DICOM) raises `ImportError`.

```python
# Default: leads with scaling metadata are converted to mV
record = FileParser.parse("ecg_file.xml")

# Raw ADC values, no conversion
record = FileParser.parse("ecg_file.xml", auto_scale=False)
```

Set to `False` to always receive raw ADC samples. See {doc}`scaling` for details on which formats provide scaling metadata and how to convert manually.

### Listing supported formats

```python
for fmt in FileParser.supported_formats():
    print(fmt["name"], fmt["extensions"])
```

### Batch parsing

`parse_batch` runs the parsers in worker processes. On macOS and Windows a
script that calls it must do so under `if __name__ == "__main__":`, otherwise
Python raises `RuntimeError` when the workers start.

```python
from ecgdatakit import parse_batch

if __name__ == "__main__":
    records = list(parse_batch(file_list, max_workers=4))

    # Keep going when a file fails, each failure is reported as BatchParseWarning
    records = list(parse_batch(file_list, on_error="warn", auto_scale=False))
```

`on_error` is `"raise"` (default), `"warn"` or `"ignore"`. Skipped files yield
nothing, so match records to files with `record.raw_metadata["filepath"]`.

```{toctree}
:hidden:

scaling
models
```

## read_ishne_annotations

```{eval-rst}
.. currentmodule:: ecgdatakit.parsing.parsers.ishne_holter

.. autofunction:: read_ishne_annotations
```

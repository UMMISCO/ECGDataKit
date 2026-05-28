# Parsing API Reference

Import: `from ecgdatakit import FileParser, parse_batch`

| | |
|---|---|
| {class}`~ecgdatakit.parsing.parser.FileParser` | Auto-detect format and parse any supported ECG file |
| {class}`~ecgdatakit.parsing.parser.Parser` | Base class for all ECG format parsers |
| {func}`~ecgdatakit.parsing.batch.parse_batch` | Parse multiple ECG files in parallel |

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
```

## Examples

### Basic usage

```python
from ecgdatakit import FileParser

fp = FileParser()
record = fp.parse("ecg_file.xml")
```

### Auto-scaling

`auto_scale` controls whether leads are automatically converted from raw ADC integers to millivolts.
When `True` (default), leads with scaling metadata (`resolution`, `offset`, `units`) are converted to **mV** automatically.
Leads without sufficient metadata are left as raw ADC values and a warning is emitted:

```
UserWarning: Leads ['Ch1', 'Ch2'] contain raw ADC samples — no scaling
metadata available. Pass auto_scale=False to get raw values.
```

```python
# Default — leads with scaling metadata are converted to mV
record = fp.parse("ecg_file.xml")

# Raw ADC values, no conversion
record = fp.parse("ecg_file.xml", auto_scale=False)
```

Set to `False` to always receive raw ADC samples. See {doc}`scaling` for details on which formats provide scaling metadata and how to convert manually.

### Listing supported formats

```python
for fmt in FileParser.supported_formats():
    print(fmt["name"], fmt["extensions"])
```

### Batch parsing

```python
from ecgdatakit import parse_batch

records = list(parse_batch(file_list, max_workers=4))
```

```{toctree}
:hidden:

scaling
models
```

# Signal Scaling

ECG files store samples as raw ADC (analog-to-digital converter) integer values.
ECGDataKit parsers read raw ADC values and attach per-lead scaling metadata so samples can be converted to physical voltage units.

## How it works

The conversion from raw ADC to physical units uses:

```{math}
\text{physical} = \text{samples} \times \text{resolution} + \text{offset}
```

Each {class}`~ecgdatakit.models.Lead` stores four scaling-related fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `resolution` | `float` | `1.0` | Multiplier — the ADC-to-physical scale factor |
| `offset` | `float` | `0.0` | Additive constant applied after scaling |
| `units` | `str` | `""` | Target physical unit, e.g. `"uV"`, `"mV"` |
| `is_raw` | `bool` | `True` | `True` if samples are raw ADC, `False` after conversion |

```{tip}
When `FileParser.parse()` is called with `auto_scale=True` (the default), leads that have scaling metadata are automatically converted to **mV**. See {doc}`parsing` for details on the `auto_scale` parameter.
```

## Format support

### Formats with scaling metadata

These formats provide per-lead scaling information — leads are auto-converted to mV when `auto_scale=True`:

| Format | Scaling source | Native unit |
|--------|---------------|-------------|
| HL7 aECG | `scale` attribute per sequence | uV |
| DICOM Waveform | Channel sensitivity and baseline | uV |
| EDF / EDF+ | Physical min/max and digital min/max | per channel |
| WFDB | Signal gain and baseline | per signal |
| SCP-ECG | AVM (amplitude value multiplier) | mV |
| ISHNE Holter | Amplitude resolution in nanovolts | mV |
| MFER | Resolution tag | per channel |
| GE MUSE XML | Waveform scale factor | uV |

### Formats without scaling metadata

These formats do not include scaling information — samples remain as raw ADC integers with `resolution=1.0`, `offset=0.0`, `units=""`:

| Format |
|--------|
| BeneHeart R12 |
| GE MAC 2000 |
| Mortara EL250 |
| Sierra XML |

## Manual conversion

### Record-level

Convert all leads at once using {meth}`~ecgdatakit.models.ECGRecord.to_physical` and {meth}`~ecgdatakit.models.ECGRecord.convert_units`:

```python
from ecgdatakit import FileParser

fp = FileParser()

# Parse with raw ADC values
record = fp.parse("ecg_file.xml", auto_scale=False)
record.leads[0].is_raw   # True

# Step 1: raw ADC → physical units (applies resolution + offset)
record = record.to_physical()
record.leads[0].is_raw   # False
record.leads[0].units    # "uV" (depends on format)

# Step 2: convert to millivolts
record = record.convert_units("mV")
record.leads[0].units    # "mV"
```

### Lead-level

Convert individual leads with {meth}`~ecgdatakit.models.Lead.to_physical` and {meth}`~ecgdatakit.models.Lead.convert_units`:

```python
lead = record.leads[0]
lead = lead.to_physical()         # raw ADC → physical
lead = lead.convert_units("mV")   # uV → mV
```

```{note}
Both methods return **new** objects — the originals are never modified.
```

### Accepted unit strings

The following aliases are recognized (case-insensitive):

| Unit | Aliases |
|------|---------|
| Microvolt | `"uV"`, `"µV"`, `"microvolt"` |
| Millivolt | `"mV"`, `"millivolt"` |
| Volt | `"V"`, `"volt"` |

### Error handling

```{warning}
Calling `convert_units()` on a lead that is still raw ADC (`is_raw=True`) raises {class}`~ecgdatakit.exceptions.RawSamplesError`. Always call `to_physical()` first.
```

```python
from ecgdatakit.exceptions import RawSamplesError

try:
    lead.convert_units("mV")
except RawSamplesError:
    lead = lead.to_physical().convert_units("mV")
```

Calling `to_physical()` on a lead with `resolution=0.0` raises `ValueError`.

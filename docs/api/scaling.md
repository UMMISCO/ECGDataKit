# Signal Scaling

ECG files store samples as raw ADC (analog-to-digital converter) integer values.
ECGDataKit parsers read raw ADC values and attach per-lead scaling metadata so samples can be converted to physical voltage units.

## How it works

The conversion from raw ADC to physical units uses:

```{math}
\text{physical} = \text{samples} \times \text{resolution} + \text{offset}
```

Each {class}`~ecgdatakit.models.Lead` stores these scaling-related fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `resolution` | `float` | `1.0` | Multiplier, the ADC-to-physical scale factor |
| `resolution_unit` | `str` | `""` | Unit of `resolution`, the unit samples are in after `to_physical()` |
| `offset` | `float` | `0.0` | Additive constant applied after scaling |
| `units` | `str` | `""` | Current unit of `samples`; empty while they are raw counts |
| `is_raw` | `bool` | `True` | `True` if samples are raw ADC, `False` after conversion |

```{tip}
When `FileParser.parse()` is called with `auto_scale=True` (the default), leads that have scaling metadata are automatically converted to **mV**. See {doc}`parsing` for details on the `auto_scale` parameter.
```

## Format support

### Formats with scaling metadata

These formats provide per-lead scaling information — leads are auto-converted to mV when `auto_scale=True`:

| Format | Scaling source | Native unit |
|--------|---------------|-------------|
| HL7 aECG | `scale` and `origin` per sequence | unit of `scale` |
| DICOM Waveform | Channel sensitivity, correction factor and baseline | UCUM unit of the channel |
| EDF / EDF+ | Physical min/max and digital min/max | per channel |
| WFDB | Signal gain and baseline | per signal |
| SCP-ECG | AVM (amplitude value multiplier, nV) | uV |
| ISHNE Holter | Amplitude resolution in nanovolts | uV |
| MFER | `MWF_SEN` (unit and exponent) and `MWF_OFF` | uV |
| GE MUSE XML | `LeadAmplitudeUnitsPerBit` and `LeadAmplitudeUnits` (a lead without `LeadAmplitudeUnits` stays raw, with a warning) | per lead |
| Mortara ELI | `UNITS_PER_MV` per channel | uV |
| Philips Sierra XML | `resolution` attribute | uV |

### Formats without scaling metadata

When a file does not state its scale, samples stay raw ADC integers with
`resolution=1.0`, `offset=0.0`, `resolution_unit=""` and `units=""`, and
`auto_scale=True` leaves them unchanged with a warning. This applies to EDAN
ARC Holter files (the count to voltage factor is not documented) and to any
file whose leads omit their scale.

## Manual conversion

`Lead.to_physical()` raises `RawSamplesError` when the file gives no physical unit. `ECGRecord.to_physical()` leaves those leads raw with a warning.

`to_dict()` and `to_json()` produce strict JSON (NaN and infinity become `null`). For long recordings use `record.to_json(fp=open(path, "w"))`, which streams with flat memory: building the string for a 24 h 12-lead Holter needs about 12 GB.

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

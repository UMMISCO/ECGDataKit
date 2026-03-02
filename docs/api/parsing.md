# Parsing API Reference

Import from the top-level package: `from ecgdatakit import FileParser, ECGRecord`

## FileParser

The main entry point. Auto-discovers all available parsers and dispatches files to the correct one based on content sniffing.

```python
from ecgdatakit import FileParser

fp = FileParser()
fp.parsers           # list of discovered Parser subclasses
record = fp.parse("ecg_file.xml")
```

| Method | Returns | Description |
|--------|---------|-------------|
| `parse(file_path)` | `ECGRecord` | Auto-detect format and parse the file |

## Parser (base class)

Abstract base class for all format-specific parsers. Located at `ecgdatakit.parsing.parser.Parser`.

| Method | Returns | Description |
|--------|---------|-------------|
| `can_parse(file_path, header)` *static* | `bool` | Return `True` if this parser handles the file |
| `parse(file_path)` | `ECGRecord` | Parse the file and return a unified record |

## parse_batch

```python
from ecgdatakit import parse_batch

records = list(parse_batch(file_list, max_workers=4))
```

Parse multiple files in parallel using `ProcessPoolExecutor`. Returns an iterator of `ECGRecord` objects.

## Data Models

All models are Python `dataclass` instances defined in `ecgdatakit.models`.

### ECGRecord

The unified output type returned by every parser.

| Field | Type | Description |
|-------|------|-------------|
| `patient` | `PatientInfo` | Patient demographics |
| `recording` | `RecordingInfo` | Recording session metadata |
| `device` | `DeviceInfo` | Acquisition device info |
| `filters` | `FilterSettings` | Filter settings applied during acquisition |
| `leads` | `list[Lead]` | ECG lead waveforms |
| `interpretation` | `Interpretation` | Machine or physician interpretation |
| `measurements` | `GlobalMeasurements` | Global ECG interval/axis measurements |
| `median_beats` | `list[Lead]` | Median/template beats if available |
| `annotations` | `dict[str, str]` | Additional key-value annotations |
| `source_format` | `str` | Parser identifier |
| `raw_metadata` | `dict` | Original format-specific metadata |

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict(include_samples=True)` | `dict` | JSON-serialisable dictionary |
| `to_json(include_samples=True, indent=2)` | `str` | JSON string |

### PatientInfo

| Field | Type | Description |
|-------|------|-------------|
| `patient_id` | `str` | Patient identifier |
| `first_name` | `str` | First name |
| `last_name` | `str` | Last name |
| `birth_date` | `datetime \| None` | Date of birth |
| `sex` | `str` | `"M"`, `"F"`, or `"U"` |
| `race` | `str` | Race/ethnicity |
| `age` | `int \| None` | Age in years |
| `weight` | `float \| None` | Weight in kg |
| `height` | `float \| None` | Height in cm |
| `medications` | `list[str]` | Current medications |
| `clinical_history` | `str` | Clinical history notes |

### RecordingInfo

| Field | Type | Description |
|-------|------|-------------|
| `date` | `datetime \| None` | Recording start time |
| `end_date` | `datetime \| None` | Recording end time |
| `duration` | `timedelta \| None` | Recording duration |
| `sample_rate` | `int` | Samples per second (Hz) |
| `adc_gain` | `float` | ADC gain factor (default 1.0) |
| `technician` | `str` | Technician name |
| `referring_physician` | `str` | Referring physician name |
| `room` | `str` | Room identifier |
| `location` | `str` | Facility/location |

### DeviceInfo

| Field | Type | Description |
|-------|------|-------------|
| `manufacturer` | `str` | Device manufacturer |
| `model` | `str` | Device model name |
| `serial_number` | `str` | Device serial number |
| `software_version` | `str` | Software version |
| `institution` | `str` | Institution name |
| `department` | `str` | Department name |
| `acquisition_type` | `str` | Acquisition type |

### FilterSettings

| Field | Type | Description |
|-------|------|-------------|
| `highpass` | `float \| None` | Highpass cutoff (Hz) |
| `lowpass` | `float \| None` | Lowpass cutoff (Hz) |
| `notch` | `float \| None` | Notch frequency (Hz) |
| `notch_active` | `bool \| None` | Whether notch filter is active |
| `artifact_filter` | `bool \| None` | Whether artifact filter is active |

### Interpretation

| Field | Type | Description |
|-------|------|-------------|
| `statements` | `list[str]` | Interpretation text statements |
| `severity` | `str` | `"NORMAL"`, `"ABNORMAL"`, `"BORDERLINE"` |
| `source` | `str` | `"machine"`, `"overread"`, `"confirmed"` |
| `interpreter` | `str` | Physician name (if overread) |
| `interpretation_date` | `datetime \| None` | When interpretation was made |

### GlobalMeasurements

| Field | Type | Description |
|-------|------|-------------|
| `heart_rate` | `int \| None` | Heart rate (bpm) |
| `rr_interval` | `int \| None` | RR interval (ms) |
| `pr_interval` | `int \| None` | PR interval (ms) |
| `qrs_duration` | `int \| None` | QRS duration (ms) |
| `qt_interval` | `int \| None` | QT interval (ms) |
| `qtc_bazett` | `int \| None` | QTc Bazett (ms) |
| `qtc_fridericia` | `int \| None` | QTc Fridericia (ms) |
| `p_axis` | `int \| None` | P-wave axis (degrees) |
| `qrs_axis` | `int \| None` | QRS axis (degrees) |
| `t_axis` | `int \| None` | T-wave axis (degrees) |
| `qrs_count` | `int \| None` | Total QRS count |

### Lead

| Field | Type | Description |
|-------|------|-------------|
| `label` | `str` | Lead name (`"I"`, `"V1"`, etc.) |
| `samples` | `NDArray[np.float64]` | Signal sample values |
| `sample_rate` | `int` | Samples per second (Hz) |
| `resolution` | `float` | Resolution (nV/unit, default 1.0) |
| `units` | `str` | Signal units (e.g. `"mV"`) |
| `quality` | `int \| None` | Signal quality indicator |
| `transducer` | `str` | Transducer type |
| `prefiltering` | `str` | Pre-filtering description |

## Working with Data Models

ECGDataKit functions accept both `Lead` objects and raw **numpy arrays**. When passing a numpy array, you must provide the sample rate via the `fs` keyword argument.

### Using numpy arrays directly

All processing and plotting functions accept numpy arrays with `fs`:

```python
import numpy as np
from ecgdatakit.processing import diagnostic_filter, detect_r_peaks
from ecgdatakit.plotting import plot_lead

signal = np.array([0.12, 0.15, 0.13, ...], dtype=np.float64)

# Pass numpy arrays with fs=
filtered = diagnostic_filter(signal, fs=500)
peaks = detect_r_peaks(filtered)
fig = plot_lead(filtered, peaks=peaks)
```

> **Note:** `fs` is required when passing a numpy array and will raise a `TypeError` if omitted. When passing a `Lead` object, `fs` is ignored.

### Using Lead objects

A `Lead` bundles samples with metadata (sample rate, label, units). Functions that return signal data always return a `Lead`:

```python
from ecgdatakit import Lead

lead = Lead(
    label="II",           # Lead name (required)
    samples=samples,      # numpy float64 array (required)
    sample_rate=500,      # Hz (required)
    units="mV",           # optional, but recommended
)

# No need for fs= when using Lead objects
filtered = diagnostic_filter(lead)
peaks = detect_r_peaks(filtered)
fig = plot_lead(filtered, peaks=peaks)
```

### Extracting numpy arrays from a Lead

To get the raw samples back (e.g. for use with scipy, sklearn, or your own code):

```python
raw_array = lead.samples          # NDArray[np.float64]
fs = lead.sample_rate             # int (Hz)
```

### Building a Lead from external data

```python
import numpy as np
from ecgdatakit import Lead

# Synthetic sine wave (10 seconds at 500 Hz)
fs = 500
t = np.arange(fs * 10, dtype=np.float64) / fs
signal = np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz ≈ 72 bpm

lead = Lead(label="II", samples=signal, sample_rate=fs, units="mV")
```

```python
# From a CSV or pandas DataFrame
import pandas as pd

df = pd.read_csv("ecg_data.csv")
lead = Lead(
    label="V1",
    samples=df["voltage"].to_numpy(dtype=np.float64),
    sample_rate=250,
    units="mV",
)
```

### Building a full ECGRecord from scratch

If you need to use functions that expect an `ECGRecord` (like `plot_12lead` or `plot_report`), you can build one manually:

```python
from ecgdatakit import ECGRecord, Lead, PatientInfo, RecordingInfo
import numpy as np

leads = [
    Lead(label=name, samples=np.random.randn(5000).astype(np.float64),
         sample_rate=500, units="mV")
    for name in ["I", "II", "III", "aVR", "aVL", "aVF",
                 "V1", "V2", "V3", "V4", "V5", "V6"]
]

record = ECGRecord(
    patient=PatientInfo(patient_id="001", first_name="Jane", last_name="Doe"),
    recording=RecordingInfo(sample_rate=500),
    leads=leads,
)
```

All fields on `ECGRecord` are optional and have sensible defaults, so you only need to provide what you have.

## Exceptions

All exceptions inherit from `ECGDataKitError`.

| Exception | When Raised |
|-----------|-------------|
| `ECGDataKitError` | Base exception for all errors |
| `UnsupportedFormatError` | File format not recognized |
| `CorruptedFileError` | File is truncated or structurally invalid |
| `MissingElementError` | Required element or field is missing |
| `ChecksumError` | Checksum validation failed |

```python
from ecgdatakit import FileParser, UnsupportedFormatError

try:
    record = FileParser().parse("unknown.bin")
except UnsupportedFormatError as e:
    print(f"Format not supported: {e}")
```

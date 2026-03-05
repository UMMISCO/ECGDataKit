# Data Models

All models are Python `dataclass` instances defined in `ecgdatakit.models`.

Import: `from ecgdatakit import ECGRecord, Lead, PatientInfo, RecordingInfo, ...`

## ECGRecord

```{eval-rst}
.. currentmodule:: ecgdatakit.models

.. autoclass:: ECGRecord
   :members:
   :undoc-members:
   :member-order: bysource
```

## Lead

```{eval-rst}
.. autoclass:: Lead
   :members:
   :undoc-members:
   :member-order: bysource
```

## PatientInfo

```{eval-rst}
.. autoclass:: PatientInfo
   :members:
   :undoc-members:
   :member-order: bysource
```

## RecordingInfo

```{eval-rst}
.. autoclass:: RecordingInfo
   :members:
   :undoc-members:
   :member-order: bysource
```

## DeviceInfo

```{eval-rst}
.. autoclass:: DeviceInfo
   :members:
   :undoc-members:
   :member-order: bysource
```

## FilterSettings

```{eval-rst}
.. autoclass:: FilterSettings
   :members:
   :undoc-members:
   :member-order: bysource
```

## AcquisitionSetup

```{eval-rst}
.. autoclass:: AcquisitionSetup
   :members:
   :undoc-members:
   :member-order: bysource
```

## SignalCharacteristics

```{eval-rst}
.. autoclass:: SignalCharacteristics
   :members:
   :undoc-members:
   :member-order: bysource
```

## Interpretation

```{eval-rst}
.. autoclass:: Interpretation
   :members:
   :undoc-members:
   :member-order: bysource
```

## GlobalMeasurements

```{eval-rst}
.. autoclass:: GlobalMeasurements
   :members:
   :undoc-members:
   :member-order: bysource
```

## Working with Data Models

ECGDataKit functions accept both {class}`~ecgdatakit.models.Lead` objects and raw **numpy arrays**. When passing a numpy array, provide the sample rate via `fs`.

### Using numpy arrays

```python
import numpy as np
from ecgdatakit.processing import diagnostic_filter, detect_r_peaks
from ecgdatakit.plotting import plot_lead

signal = np.array([0.12, 0.15, 0.13, ...], dtype=np.float64)

filtered = diagnostic_filter(signal, fs=500)
peaks = detect_r_peaks(filtered)
fig = plot_lead(filtered, peaks=peaks)
```

> **Note:** `fs` is required when passing a numpy array and will raise a `TypeError` if omitted. When passing a {class}`~ecgdatakit.models.Lead`, `fs` is ignored.

### Using Lead objects

```python
from ecgdatakit import Lead

lead = Lead(
    label="II",
    samples=samples,
    sample_rate=500,
    units="mV",
    is_raw=False,
)

# No need for fs= when using Lead objects
filtered = diagnostic_filter(lead)
```

### Extracting numpy arrays

```python
raw_array = lead.samples     # NDArray[np.float64]
fs = lead.sample_rate        # int (Hz)
```

### Building a Lead from external data

```python
import numpy as np
from ecgdatakit import Lead

# Synthetic sine wave (10 s at 500 Hz)
fs = 500
t = np.arange(fs * 10, dtype=np.float64) / fs
signal = np.sin(2 * np.pi * 1.2 * t)

lead = Lead(label="II", samples=signal, sample_rate=fs, units="mV", is_raw=False)
```

```python
# From a pandas DataFrame
import pandas as pd

df = pd.read_csv("ecg_data.csv")
lead = Lead(
    label="V1",
    samples=df["voltage"].to_numpy(dtype=np.float64),
    sample_rate=250,
    units="mV",
    is_raw=False,
)
```

### Building an ECGRecord from scratch

```python
from ecgdatakit import ECGRecord, Lead, PatientInfo, RecordingInfo
import numpy as np

leads = [
    Lead(label=name, samples=np.random.randn(5000).astype(np.float64),
         sample_rate=500, units="mV", is_raw=False)
    for name in ["I", "II", "III", "aVR", "aVL", "aVF",
                 "V1", "V2", "V3", "V4", "V5", "V6"]
]

rec = RecordingInfo()
rec.acquisition.signal.sample_rate = 500

record = ECGRecord(
    patient=PatientInfo(patient_id="001", first_name="Jane", last_name="Doe"),
    recording=rec,
    leads=leads,
)
```

All fields are optional with sensible defaults.

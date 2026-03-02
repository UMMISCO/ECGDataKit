# ECGDataKit

**A Python library for parsing, processing, and visualizing multi-format ECG files.**

## Quick Install

```bash
pip install ecgdatakit
```

## Quick Example

```python
from ecgdatakit import FileParser
from ecgdatakit.processing import diagnostic_filter, detect_r_peaks, heart_rate
from ecgdatakit.plotting import plot_12lead, plot_peaks

# Parse any ECG file (auto-detect format)
record = FileParser().parse("ecg_file.xml")

# Filter and detect R-peaks
lead = record.leads[1]  # Lead II
filtered = diagnostic_filter(lead)
peaks = detect_r_peaks(filtered)
print(f"Heart rate: {heart_rate(filtered, peaks):.0f} bpm")

# Visualize
fig = plot_12lead(record)
fig.savefig("ecg_report.png", dpi=150)
```

## Features

- **Multi-Format Parsing** -- Parse HL7 aECG, Philips Sierra, GE MUSE, SCP-ECG, DICOM, EDF, WFDB, MFER, and more into one unified data structure.
- **Signal Processing** -- Butterworth filters, R-peak detection, HRV analysis, FFT, signal quality, lead derivation, ECG cleaning, and DeepFADE neural-net denoising.
- **Visualization** -- Standard 12-lead grids, R-peak annotations, HRV dashboards, spectrograms, and full ECG reports. Static (matplotlib) or interactive (plotly).

```{toctree}
:maxdepth: 2
:caption: Guides

guides/getting-started
guides/formats
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/parsing
api/processing
api/plotting
api/reference
api/models
api/exceptions
```

```{toctree}
:maxdepth: 1
:caption: Project

releases
```

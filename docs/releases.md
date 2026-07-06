# Releases

All published releases of ECGDataKit, fetched from [GitHub Releases](https://github.com/UMMISCO/ECGDataKit/releases) at build time.

---

(v1-0-2)=
## v1.0.2

*Released 2026-07-06* &middot; [View on GitHub](https://github.com/UMMISCO/ECGDataKit/releases/tag/v1.0.2)

**Full Changelog**: https://github.com/UMMISCO/ECGDataKit/compare/v1.0.0...v1.0.2

(v1-0-0)=
## v1.0.0 - Stable Release

*Released 2026-03-19* &middot; [View on GitHub](https://github.com/UMMISCO/ECGDataKit/releases/tag/v1.0.0)

### New features

- **Per-lead normalization** — `normalize_minmax`, `normalize_zscore`, and `normalize_amplitude` now accept a `list[Lead]` and return a `list[Lead]`, normalizing each lead independently. Pass `record.leads` directly instead of looping manually
- Resampling and normalization (min-max, z-score, amplitude) now support batch processing across all leads (independently) of an ECG record


**Full Changelog**: https://github.com/UMMISCO/ECGDataKit/compare/v0.0.9...v1.0.0

(v0-0-9)=
## v0.0.9 - ADC resolution pipeline

*Released 2026-03-07* &middot; [View on GitHub](https://github.com/UMMISCO/ECGDataKit/releases/tag/v0.0.9)

### Breaking changes

  - **`sample_rate` renamed to `sampling_rate`** in `Lead` and `SignalCharacteristics`
  - **`Lead.units` semantics changed** — now empty (`""`) when `is_raw=True` (raw ADC counts); set to the physical unit (voltages) only after `to_physical()` or when data is already in physical units
  - **`Lead.is_raw` is now auto-detected** — parsers no longer hardcode `is_raw=True`; instead `is_raw = not (resolution == 1.0 and offset == 0.0)`

  ### New features

  - **Automatic ADC-to-physical scaling** — `FileParser.parse(auto_scale=True)` (default) converts raw ADC samples to mV via `to_physical()` + `convert_units("mV")`. Disable with `auto_scale=False`
  - **`Lead.to_physical()`** — converts raw ADC samples using `physical = samples × resolution + offset`
  - **`Lead.convert_units(target)`** — converts between voltage units (uV, mV, V)
  - **`ECGRecord.to_physical()` / `ECGRecord.convert_units(target)`** — batch conversion for all leads and median beats
  - **`FileParser.supported_formats()`** — returns format metadata for all 12 parsers
  - **`ECGRecord.__repr__()` and `ECGRecord.plot()`** — YAML-style console display and quick plotting
  - **Multi-lead numpy array support** — `plot_leads`, `plot_12lead`, `iplot_leads`, `iplot_12lead` accept raw numpy arrays with `fs=` parameter
  - **`LeadsLike` type alias** — for multi-lead inputs (list of Lead, ECGRecord, 2D array, list of arrays)

  ### New Lead fields

  - **`resolution_unit`** — unit of the resolution scale factor (e.g. `"uV"`, `"mV"`); what samples will be in after `to_physical()`
  - **`adc_resolution`** — original ADC resolution as stored in the source file (e.g. `153.0` for 153 nV/count in ISHNE)
  - **`adc_resolution_unit`** — unit of `adc_resolution` as defined by the format (e.g. `"nV"` for ISHNE and SCP-ECG)

  ### Improvements

  - Simplified multi-lead plots: full signal by default with configurable `rows`/`cols` grid layout
  - `plot_12lead` and `iplot_12lead` assign standard 12-lead names to unnamed inputs
  - Static plots no longer force matplotlib Agg backend
  

**Full Changelog**: https://github.com/UMMISCO/ECGDataKit/commits/v0.0.9

(v0-0-8)=
## v0.0.8 - Visualization

*Released 2026-03-03* &middot; [View on GitHub](https://github.com/UMMISCO/ECGDataKit/releases/tag/v0.0.8)

### Visualization

- Multi-lead plotting functions (`plot_leads`, `plot_12lead`, `plot_quality`, `iplot_leads`, `iplot_12lead`) now accept raw numpy arrays directly with `fs=` parameter — pass a 2D array (n_leads × n_samples) or a list of 1D arrays alongside the sample rate
- New `LeadsLike` type alias for multi-lead inputs: `list[Lead] | ECGRecord | NDArray | list[NDArray]`
- Static plots no longer force the `Agg` backend — plots display inline in Jupyter and GUI environments by default


**Full Changelog**: https://github.com/UMMISCO/ECGDataKit/commits/v0.0.8

(v0-0-0-7)=
## v0.0.7 - Signal Characteristics, Flexible Inputs & Sphinx Docs

*Released 2026-03-02* &middot; [View on GitHub](https://github.com/UMMISCO/ECGDataKit/releases/tag/v0.0.0.7)

## What's New

  ### Signal Characteristics Model

  New `SignalCharacteristics` dataclass on every `ECGRecord` — captures technical signal encoding metadata directly from each format:

  - `bits_per_sample`, `signal_offset`, `signal_signed`
  - `data_encoding` (e.g. `base64_int16le`, `format_212`, `int16`)
  - `compression` (e.g. `none`, `huffman`)
  - `number_channels_allocated` / `number_channels_valid`
  - `electrode_placement`, `acsetting`, and signal processing flags

  All 12 parsers now populate `record.signal` from format-specific fields.

  ### Expanded Metadata Extraction

  Parsers now extract significantly more metadata from each format:

  - **Technician** — Sierra XML, GE MUSE, DICOM, GE MAC 2000
  - **Referring physician** — Sierra XML, EDF, GE MUSE, GE MAC 2000
  - **Patient weight** — SCP-ECG (Tag 12), BeneHeart R12, GE MAC 2000
  - **Patient height** — BeneHeart R12, GE MAC 2000
  - **Patient race** — HL7 aECG, Mortara EL250, GE MAC 2000
  - **RR interval & QRS count** — Mortara EL250, BeneHeart R12, GE MAC 2000
  - **Device serial number** — HL7 aECG, GE MAC 2000
  - **Notch filter active flag** — now properly set across all parsers

  ### Flexible Inputs — Numpy Arrays Everywhere

  All processing and plotting functions now accept **raw numpy arrays** directly alongside `Lead` objects. Pass the sample rate via `fs=`:

  ```python
  import numpy as np
  from ecgdatakit.processing import diagnostic_filter, detect_r_peaks
  from ecgdatakit.plotting import plot_lead

  signal = np.random.randn(5000)
  filtered = diagnostic_filter(signal, fs=500)
  peaks = detect_r_peaks(filtered)
  plot_lead(filtered, peaks=peaks)
```

  ### Plotting Improvements

  - Auto-display: static plots now call plt.show() by default — pass show=False to get the figure for saving
  - Sample index axis: plot_lead(lead, x_axis="samples") for sample indices instead of time
  - Cleaner layout: reduced margins, integer time ticks on x-axis

  ### Documentation Migration

  - Migrated from Hugo to Sphinx with the Furo theme
  - Full autodoc API reference generated from source docstrings
  - Hosted at https://ecgdatakit.ummisco.fr

  Bug Fixes

  - BeneHeart R12: fixed assignment to non-existent RecordingInfo.device field
  - SCP-ECG: now reads patient weight from Tag 12 (was silently skipped)
  - MFER: notch filter now correctly extracted from filter tag when 12+ bytes present


**Full Changelog**: https://github.com/UMMISCO/ECGDataKit/commits/v0.0.0.7

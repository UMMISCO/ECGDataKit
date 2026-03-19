# Processing API Reference

Import: `from ecgdatakit.processing import ...`

Requires: `pip install ecgdatakit[processing]` (scipy ≥ 1.10)

All filter and transform functions accept a {class}`~ecgdatakit.models.Lead` and return a **new** {class}`~ecgdatakit.models.Lead` (immutable pattern via `dataclasses.replace`). The original lead is never modified.

```{note}
All processing functions accept both `Lead` objects and raw numpy arrays. When passing a numpy array, provide the sample rate via `fs`:

    filtered = diagnostic_filter(my_array, fs=500)

See {doc}`models` for details.
```

## {doc}`Filters <processing/filters>`

| | |
|---|---|
| {func}`~ecgdatakit.processing.lowpass` | Apply a Butterworth low-pass filter |
| {func}`~ecgdatakit.processing.highpass` | Apply a Butterworth high-pass filter |
| {func}`~ecgdatakit.processing.bandpass` | Apply a Butterworth band-pass filter |
| {func}`~ecgdatakit.processing.notch` | Apply an IIR notch (band-stop) filter |
| {func}`~ecgdatakit.processing.remove_baseline` | Remove baseline wander using a high-pass filter |
| {func}`~ecgdatakit.processing.diagnostic_filter` | AHA diagnostic: 0.05–150 Hz bandpass + notch |
| {func}`~ecgdatakit.processing.monitoring_filter` | Monitoring: 0.67–40 Hz bandpass + notch |

## {doc}`Resampling <processing/resampling>`

| | |
|---|---|
| {func}`~ecgdatakit.processing.resample` | Resample a lead to a different sample rate |

## {doc}`Normalization <processing/normalization>`

All normalization functions accept a single lead **or** a `list[Lead]` for per-lead normalization.

| | |
|---|---|
| {func}`~ecgdatakit.processing.normalize_minmax` | Scale signal to the [−1, 1] range |
| {func}`~ecgdatakit.processing.normalize_zscore` | Normalize to zero mean and unit variance (z-score) |
| {func}`~ecgdatakit.processing.normalize_amplitude` | Scale peak amplitude to a target value |

## {doc}`R-Peak Detection <processing/peaks>`

| | |
|---|---|
| {func}`~ecgdatakit.processing.detect_r_peaks` | Detect R-peak locations in an ECG lead |
| {func}`~ecgdatakit.processing.heart_rate` | Compute average heart rate in beats per minute |
| {func}`~ecgdatakit.processing.rr_intervals` | Compute RR intervals in milliseconds |
| {func}`~ecgdatakit.processing.instantaneous_heart_rate` | Compute instantaneous heart rate at each beat |

## {doc}`Heart Rate Variability <processing/hrv>`

| | |
|---|---|
| {func}`~ecgdatakit.processing.time_domain` | Compute time-domain HRV metrics from RR intervals |
| {func}`~ecgdatakit.processing.frequency_domain` | Compute frequency-domain HRV metrics from RR intervals |
| {func}`~ecgdatakit.processing.poincare` | Compute Poincaré plot descriptors (SD1, SD2) |

## {doc}`Transforms & Segmentation <processing/transforms>`

| | |
|---|---|
| {func}`~ecgdatakit.processing.power_spectrum` | Compute the power spectral density of an ECG lead |
| {func}`~ecgdatakit.processing.fft` | Compute the single-sided FFT magnitude spectrum |
| {func}`~ecgdatakit.processing.segment_beats` | Segment individual heartbeats around R-peaks |
| {func}`~ecgdatakit.processing.average_beat` | Compute the ensemble-averaged heartbeat (template) |

## {doc}`Signal Quality <processing/quality>`

| | |
|---|---|
| {func}`~ecgdatakit.processing.signal_quality_index` | Composite signal quality index (SQI) in [0, 1] |
| {func}`~ecgdatakit.processing.classify_quality` | Classify signal quality as a human-readable category |
| {func}`~ecgdatakit.processing.snr_estimate` | Estimate signal-to-noise ratio in dB |

## {doc}`Lead Derivation <processing/leads>`

| | |
|---|---|
| {func}`~ecgdatakit.processing.derive_lead_iii` | Derive Lead III from Leads I and II (Einthoven's law) |
| {func}`~ecgdatakit.processing.derive_augmented` | Derive augmented limb leads aVR, aVL, aVF |
| {func}`~ecgdatakit.processing.derive_standard_12` | Assemble a full 12-lead ECG |
| {func}`~ecgdatakit.processing.find_lead` | Find a lead by label (case-insensitive) |

## {doc}`ECG Cleaning <processing/cleaning>`

| | |
|---|---|
| {func}`~ecgdatakit.processing.clean_ecg` | Clean an ECG lead signal |

```{toctree}
:hidden:

processing/filters
processing/resampling
processing/normalization
processing/peaks
processing/hrv
processing/transforms
processing/quality
processing/leads
processing/cleaning
```

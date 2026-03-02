---
title: "Processing API Reference"
weight: 20
---

Import: `from ecgdatakit.processing import ...`

Requires: `pip install ecgdatakit[processing]` (scipy ≥ 1.10)

All filter and transform functions accept a `Lead` and return a **new** `Lead` (immutable pattern via `dataclasses.replace`). The original lead is never modified.

## Filters

All filters use SOS (second-order sections) + zero-phase `sosfiltfilt` to preserve ECG morphology.

<table>
  <thead><tr><th>Function</th><th>Returns</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lowpass(lead, cutoff, order=4)</code></td><td><code>Lead</code></td><td>Butterworth low-pass filter</td></tr>
    <tr><td><code>highpass(lead, cutoff, order=4)</code></td><td><code>Lead</code></td><td>Butterworth high-pass filter</td></tr>
    <tr><td><code>bandpass(lead, low, high, order=4)</code></td><td><code>Lead</code></td><td>Butterworth band-pass filter</td></tr>
    <tr><td><code>notch(lead, freq=50.0, quality=30.0)</code></td><td><code>Lead</code></td><td>IIR notch (band-stop) filter</td></tr>
    <tr><td><code>remove_baseline(lead, cutoff=0.5, order=2)</code></td><td><code>Lead</code></td><td>Remove baseline wander (highpass at 0.5 Hz)</td></tr>
    <tr><td><code>diagnostic_filter(lead, notch_freq=50.0)</code></td><td><code>Lead</code></td><td>AHA diagnostic: 0.05–150 Hz bandpass + notch</td></tr>
    <tr><td><code>monitoring_filter(lead, notch_freq=50.0)</code></td><td><code>Lead</code></td><td>Monitoring: 0.67–40 Hz bandpass + notch</td></tr>
  </tbody>
</table>

```python
from ecgdatakit.processing import diagnostic_filter, notch

# Diagnostic-grade filtering
filtered = diagnostic_filter(lead)

# Or build a custom pipeline
filtered = notch(lead, freq=60.0)  # 60 Hz for US mains
```

## Resampling

<table>
  <thead><tr><th>Function</th><th>Returns</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>resample(lead, target_rate)</code></td><td><code>Lead</code></td><td>Polyphase resampling to target sample rate</td></tr>
  </tbody>
</table>

```python
from ecgdatakit.processing import resample

lead_250 = resample(lead, target_rate=250)
print(lead_250.sample_rate)  # 250
```

## Normalization

Pure numpy — no scipy required.

<table>
  <thead><tr><th>Function</th><th>Returns</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>normalize_minmax(lead)</code></td><td><code>Lead</code></td><td>Scale to [−1, 1] range</td></tr>
    <tr><td><code>normalize_zscore(lead)</code></td><td><code>Lead</code></td><td>Zero mean, unit variance</td></tr>
    <tr><td><code>normalize_amplitude(lead, target_mv=1.0)</code></td><td><code>Lead</code></td><td>Scale peak amplitude to target</td></tr>
  </tbody>
</table>

## R-Peak Detection

<table>
  <thead><tr><th>Function</th><th>Returns</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>detect_r_peaks(lead, method="pan_tompkins")</code></td><td><code>NDArray[np.intp]</code></td><td>R-peak sample indices. Methods: <code>"pan_tompkins"</code> (bandpass + derivative + adaptive threshold) or <code>"shannon_energy"</code> (Shannon energy envelope detector).</td></tr>
    <tr><td><code>heart_rate(lead, peaks=None)</code></td><td><code>float</code></td><td>Average heart rate in bpm</td></tr>
    <tr><td><code>rr_intervals(lead, peaks=None)</code></td><td><code>NDArray[np.float64]</code></td><td>RR intervals in milliseconds</td></tr>
    <tr><td><code>instantaneous_heart_rate(lead, peaks=None)</code></td><td><code>NDArray[np.float64]</code></td><td>Beat-by-beat heart rate in bpm</td></tr>
  </tbody>
</table>

All functions auto-detect peaks if `peaks=None`.

```python
from ecgdatakit.processing import detect_r_peaks, heart_rate, rr_intervals

# Pan-Tompkins (default)
peaks = detect_r_peaks(filtered)

# Shannon energy envelope — good for noisy signals
peaks_se = detect_r_peaks(filtered, method="shannon_energy")

hr = heart_rate(filtered, peaks)      # e.g. 72.5
rr = rr_intervals(filtered, peaks)    # array of RR in ms
```

## Heart Rate Variability

<table>
  <thead><tr><th>Function</th><th>Returns</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code>time_domain(rr_ms)</code></td>
      <td><code>dict</code></td>
      <td>Keys: <code>mean_rr</code>, <code>sdnn</code>, <code>rmssd</code>, <code>sdsd</code>, <code>nn50_count</code>, <code>pnn50</code>, <code>nn20_count</code>, <code>pnn20</code>, <code>hr_mean</code>, <code>hr_std</code></td>
    </tr>
    <tr>
      <td><code>frequency_domain(rr_ms, method="welch", interp_fs=4.0)</code></td>
      <td><code>dict</code></td>
      <td>Keys: <code>vlf_power</code> (0–0.04 Hz), <code>lf_power</code> (0.04–0.15 Hz), <code>hf_power</code> (0.15–0.4 Hz), <code>lf_hf_ratio</code>, <code>total_power</code></td>
    </tr>
    <tr>
      <td><code>poincare(rr_ms)</code></td>
      <td><code>dict</code></td>
      <td>Keys: <code>sd1</code>, <code>sd2</code>, <code>sd1_sd2_ratio</code></td>
    </tr>
  </tbody>
</table>

```python
from ecgdatakit.processing import time_domain, frequency_domain, poincare

hrv_t = time_domain(rr)
print(hrv_t["sdnn"], hrv_t["rmssd"], hrv_t["pnn50"])

hrv_f = frequency_domain(rr)
print(hrv_f["lf_hf_ratio"])

p = poincare(rr)
print(p["sd1"], p["sd2"])
```

## Transforms & Segmentation

<table>
  <thead><tr><th>Function</th><th>Returns</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>power_spectrum(lead, method="welch", nperseg=None)</code></td><td><code>tuple[NDArray, NDArray]</code></td><td>(frequencies, power) — Welch PSD</td></tr>
    <tr><td><code>fft(lead)</code></td><td><code>tuple[NDArray, NDArray]</code></td><td>(frequencies, magnitudes) — single-sided FFT</td></tr>
    <tr><td><code>segment_beats(lead, peaks=None, before=0.2, after=0.4)</code></td><td><code>list[Lead]</code></td><td>Extract individual heartbeats around R-peaks</td></tr>
    <tr><td><code>average_beat(lead, peaks=None, before=0.2, after=0.4)</code></td><td><code>Lead</code></td><td>Ensemble-averaged beat template</td></tr>
  </tbody>
</table>

## Signal Quality

<table>
  <thead><tr><th>Function</th><th>Returns</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>signal_quality_index(lead)</code></td><td><code>float</code></td><td>Composite SQI score (0.0–1.0)</td></tr>
    <tr><td><code>classify_quality(lead)</code></td><td><code>str</code></td><td><code>"excellent"</code> (&gt;0.8), <code>"acceptable"</code> (0.5–0.8), <code>"unacceptable"</code> (&lt;0.5)</td></tr>
    <tr><td><code>snr_estimate(lead)</code></td><td><code>float</code></td><td>Signal-to-noise ratio in dB</td></tr>
  </tbody>
</table>

## Lead Derivation

Pure numpy — no scipy required. Validates sample rate and length match.

<table>
  <thead><tr><th>Function</th><th>Returns</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>derive_lead_iii(lead_i, lead_ii)</code></td><td><code>Lead</code></td><td>III = II − I (Einthoven's law)</td></tr>
    <tr><td><code>derive_augmented(lead_i, lead_ii)</code></td><td><code>list[Lead]</code></td><td>[aVR, aVL, aVF]</td></tr>
    <tr><td><code>derive_standard_12(lead_i, lead_ii, v1..v6)</code></td><td><code>list[Lead]</code></td><td>Full 12-lead set in standard order</td></tr>
    <tr><td><code>find_lead(leads, label)</code></td><td><code>Lead | None</code></td><td>Case-insensitive lead lookup</td></tr>
  </tbody>
</table>

```python
from ecgdatakit.processing import derive_augmented, find_lead

lead_i = find_lead(record.leads, "I")
lead_ii = find_lead(record.leads, "II")

avr, avl, avf = derive_augmented(lead_i, lead_ii)
```

## ECG Cleaning

Unified cleaning interface with multiple backends. All methods return a new `Lead`.

<table>
  <thead><tr><th>Function</th><th>Returns</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>clean_ecg(lead, method="default", **kwargs)</code></td><td><code>Lead</code></td><td>Clean an ECG lead. Methods: <code>"default"</code>, <code>"biosppy"</code>, <code>"neurokit2"</code>, <code>"combined"</code>, <code>"deepfade"</code>.</td></tr>
  </tbody>
</table>

<table>
  <thead><tr><th>Method</th><th>Extra dependency</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>"default"</code></td><td>scipy</td><td>Bandpass 0.5–40 Hz + 50 Hz notch</td></tr>
    <tr><td><code>"biosppy"</code></td><td><code>pip install biosppy</code></td><td>BioSPPy ECG filter</td></tr>
    <tr><td><code>"neurokit2"</code></td><td><code>pip install neurokit2</code></td><td>NeuroKit2 adaptive pipeline</td></tr>
    <tr><td><code>"combined"</code></td><td>biosppy + neurokit2</td><td>BioSPPy → NeuroKit2</td></tr>
    <tr><td><code>"deepfade"</code></td><td><code>pip install torch</code></td><td>DeepFADE DenseNet encoder-decoder (weights bundled)</td></tr>
  </tbody>
</table>

```python
from ecgdatakit.processing import clean_ecg

cleaned = clean_ecg(lead)
cleaned = clean_ecg(lead, method="neurokit2")
cleaned = clean_ecg(lead, method="deepfade")
cleaned = clean_ecg(lead, method="deepfade", device="mps")
```

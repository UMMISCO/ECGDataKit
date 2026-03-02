---
title: "snr_estimate"
weight: 26
---

`ecgdatakit.processing.quality.snr_estimate`

**Module:** Signal Quality

## Signature

```python
snr_estimate(lead, *, fs=None)
```

## Description

Estimate signal-to-noise ratio in dB.

Uses a frequency-domain approach: signal power in 1--40 Hz
vs noise power above 100 Hz (up to Nyquist).

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>Input ECG lead or raw signal array.</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Returns

<table>
  <thead><tr><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>float</code></td><td>Estimated SNR in decibels.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import snr_estimate

result = snr_estimate(...)
```

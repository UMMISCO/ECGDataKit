---
title: "instantaneous_heart_rate"
weight: 16
---

`ecgdatakit.processing.peaks.instantaneous_heart_rate`

**Module:** R-Peak Detection

## Signature

```python
instantaneous_heart_rate(lead, peaks=None, *, fs=None)
```

## Description

Compute instantaneous heart rate at each beat in bpm.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>Input ECG lead or raw signal array.</td></tr>
    <tr><td><code>peaks</code></td><td><code>NDArray | None</code></td><td>Pre-detected R-peak indices.  Detected automatically if ``None``.</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import instantaneous_heart_rate

result = instantaneous_heart_rate(...)
```

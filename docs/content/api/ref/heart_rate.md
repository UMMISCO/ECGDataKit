---
title: "heart_rate"
weight: 14
---

`ecgdatakit.processing.peaks.heart_rate`

**Module:** R-Peak Detection

## Signature

```python
heart_rate(lead, peaks=None, *, fs=None)
```

## Description

Compute average heart rate in beats per minute.

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
from ecgdatakit.processing import heart_rate

result = heart_rate(...)
```

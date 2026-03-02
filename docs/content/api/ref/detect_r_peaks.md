---
title: "detect_r_peaks"
weight: 13
---

`ecgdatakit.processing.peaks.detect_r_peaks`

**Module:** R-Peak Detection

## Signature

```python
detect_r_peaks(lead, method='pan_tompkins', *, fs=None)
```

## Description

Detect R-peak locations in an ECG lead.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>Input ECG lead or raw signal array (typically Lead II).</td></tr>
    <tr><td><code>method</code></td><td><code>str</code></td><td>Detection algorithm: ``&quot;pan_tompkins&quot;`` or ``&quot;shannon_energy&quot;``.</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Returns

<table>
  <thead><tr><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>NDArray[np.intp]</code></td><td>Array of sample indices where R-peaks were detected.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import detect_r_peaks

result = detect_r_peaks(...)
```

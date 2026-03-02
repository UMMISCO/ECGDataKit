---
title: "diagnostic_filter"
weight: 7
---

`ecgdatakit.processing.filters.diagnostic_filter`

**Module:** Filters

## Signature

```python
diagnostic_filter(lead, notch_freq=50.0, *, fs=None)
```

## Description

Apply AHA diagnostic-grade filtering: 0.05–150 Hz bandpass + notch.

Suitable for diagnostic ECG interpretation where full morphology
(including ST segment) must be preserved.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>Input ECG lead or raw signal array.</td></tr>
    <tr><td><code>notch_freq</code></td><td><code>float</code></td><td>Power-line frequency to notch out (50 or 60 Hz).</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import diagnostic_filter

result = diagnostic_filter(...)
```

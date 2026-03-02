---
title: "bandpass"
weight: 4
---

`ecgdatakit.processing.filters.bandpass`

**Module:** Filters

## Signature

```python
bandpass(lead, low, high, order=4, *, fs=None)
```

## Description

Apply a Butterworth band-pass filter.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>Input ECG lead or raw signal array.</td></tr>
    <tr><td><code>low</code></td><td><code>float</code></td><td>Lower cutoff frequency in Hz.</td></tr>
    <tr><td><code>high</code></td><td><code>float</code></td><td>Upper cutoff frequency in Hz.</td></tr>
    <tr><td><code>order</code></td><td><code>int</code></td><td>Filter order (default 4).</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import bandpass

result = bandpass(...)
```

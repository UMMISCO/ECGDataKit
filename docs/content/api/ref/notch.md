---
title: "notch"
weight: 5
---

`ecgdatakit.processing.filters.notch`

**Module:** Filters

## Signature

```python
notch(lead, freq=50.0, quality=30.0, *, fs=None)
```

## Description

Apply an IIR notch (band-stop) filter.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>Input ECG lead or raw signal array.</td></tr>
    <tr><td><code>freq</code></td><td><code>float</code></td><td>Center frequency to remove (default 50 Hz for mains hum).</td></tr>
    <tr><td><code>quality</code></td><td><code>float</code></td><td>Quality factor — higher means narrower notch (default 30).</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import notch

result = notch(...)
```

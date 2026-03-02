---
title: "fft"
weight: 21
---

`ecgdatakit.processing.transforms.fft`

**Module:** Transforms & Segmentation

## Signature

```python
fft(lead, *, fs=None)
```

## Description

Compute the single-sided FFT magnitude spectrum.

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
    <tr><td><code>tuple[NDArray, NDArray]</code></td><td>``(frequencies, magnitudes)`` arrays (positive frequencies only).</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import fft

result = fft(...)
```

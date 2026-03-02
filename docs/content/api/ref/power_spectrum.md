---
title: "power_spectrum"
weight: 20
---

`ecgdatakit.processing.transforms.power_spectrum`

**Module:** Transforms & Segmentation

## Signature

```python
power_spectrum(lead, method='welch', nperseg=None, *, fs=None)
```

## Description

Compute the power spectral density of an ECG lead.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>Input ECG lead or raw signal array.</td></tr>
    <tr><td><code>method</code></td><td><code>str</code></td><td>``&quot;welch&quot;`` (default) for Welch&#x27;s method.</td></tr>
    <tr><td><code>nperseg</code></td><td><code>int | None</code></td><td>Segment length for Welch&#x27;s method. Defaults to ``min(256, len(samples))``.</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Returns

<table>
  <thead><tr><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>tuple[NDArray, NDArray]</code></td><td>``(frequencies, power)`` arrays.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import power_spectrum

result = power_spectrum(...)
```

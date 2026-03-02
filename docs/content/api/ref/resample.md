---
title: "resample"
weight: 9
---

`ecgdatakit.processing.resample.resample`

**Module:** Resampling

## Signature

```python
resample(lead, target_rate, *, fs=None)
```

## Description

Resample a lead to a different sample rate.

Uses polyphase rational resampling (``scipy.signal.resample_poly``)
which preserves signal content up to the Nyquist frequency of the
lower sample rate.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>Input ECG lead or raw signal array.</td></tr>
    <tr><td><code>target_rate</code></td><td><code>int</code></td><td>Desired output sample rate in Hz.</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import resample

result = resample(...)
```

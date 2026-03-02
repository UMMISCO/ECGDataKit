---
title: "plot_spectrogram"
weight: 38
---

`ecgdatakit.plotting.static.plot_spectrogram`

**Module:** Static Plots (matplotlib)

## Signature

```python
plot_spectrogram(lead, nperseg=256, figsize=(12, 4), ax=None, *, fs=None)
```

## Description

Plot time-frequency spectrogram (STFT).

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>ECG lead or raw signal array.</td></tr>
    <tr><td><code>nperseg</code></td><td><code>int</code></td><td>Segment length for STFT.</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.plotting import plot_spectrogram

result = plot_spectrogram(...)
```

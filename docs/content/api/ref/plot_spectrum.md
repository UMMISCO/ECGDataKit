---
title: "plot_spectrum"
weight: 37
---

`ecgdatakit.plotting.static.plot_spectrum`

**Module:** Static Plots (matplotlib)

## Signature

```python
plot_spectrum(lead, method='welch', figsize=(10, 4), ax=None, *, fs=None)
```

## Description

Plot power spectral density or FFT magnitude spectrum.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>ECG lead or raw signal array to analyse.</td></tr>
    <tr><td><code>method</code></td><td><code>str</code></td><td>``&quot;welch&quot;`` for PSD or ``&quot;fft&quot;`` for magnitude spectrum.</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.plotting import plot_spectrum

result = plot_spectrum(...)
```

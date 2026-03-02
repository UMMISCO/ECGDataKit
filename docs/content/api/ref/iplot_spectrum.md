---
title: "iplot_spectrum"
weight: 48
---

`ecgdatakit.plotting.interactive.iplot_spectrum`

**Module:** Interactive Plots (plotly)

## Signature

```python
iplot_spectrum(lead, method='welch', height=400, *, fs=None)
```

## Description

Interactive spectrum with frequency band highlighting.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>ECG lead or raw signal array.</td></tr>
    <tr><td><code>method</code></td><td><code>str</code></td><td>``&quot;welch&quot;`` for PSD or ``&quot;fft&quot;`` for magnitude spectrum.</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.plotting import iplot_spectrum

result = iplot_spectrum(...)
```

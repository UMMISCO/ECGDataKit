---
title: "iplot_peaks"
weight: 47
---

`ecgdatakit.plotting.interactive.iplot_peaks`

**Module:** Interactive Plots (plotly)

## Signature

```python
iplot_peaks(lead, peaks=None, title=None, height=350, *, fs=None)
```

## Description

Interactive lead with R-peak markers.

Hover on peaks shows peak index, RR interval, and instantaneous HR.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>ECG lead or raw signal array to plot.</td></tr>
    <tr><td><code>peaks</code></td><td><code>NDArray | None</code></td><td>R-peak indices. Auto-detected if ``None``.</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.plotting import iplot_peaks

result = iplot_peaks(...)
```

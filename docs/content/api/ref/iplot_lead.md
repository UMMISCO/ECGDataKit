---
title: "iplot_lead"
weight: 44
---

`ecgdatakit.plotting.interactive.iplot_lead`

**Module:** Interactive Plots (plotly)

## Signature

```python
iplot_lead(lead, peaks=None, title=None, height=300, *, fs=None)
```

## Description

Interactive single lead with hover showing time/amplitude.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>ECG lead or raw signal array to plot.</td></tr>
    <tr><td><code>peaks</code></td><td><code>NDArray | None</code></td><td>Optional R-peak indices to mark.</td></tr>
    <tr><td><code>title</code></td><td><code>str | None</code></td><td>Figure title.</td></tr>
    <tr><td><code>height</code></td><td><code>int</code></td><td>Figure height in pixels.</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.plotting import iplot_lead

result = iplot_lead(...)
```

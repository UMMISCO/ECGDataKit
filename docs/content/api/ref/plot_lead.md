---
title: "plot_lead"
weight: 31
---

`ecgdatakit.plotting.static.plot_lead`

**Module:** Static Plots (matplotlib)

## Signature

```python
plot_lead(lead, peaks=None, title=None, show_grid=True, figsize=(12, 3), ax=None, *, fs=None)
```

## Description

Plot a single ECG lead waveform.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>ECG lead or raw signal array to plot.</td></tr>
    <tr><td><code>peaks</code></td><td><code>NDArray | None</code></td><td>Optional R-peak indices to mark.</td></tr>
    <tr><td><code>title</code></td><td><code>str | None</code></td><td>Figure title. Defaults to the lead label.</td></tr>
    <tr><td><code>show_grid</code></td><td><code>bool</code></td><td>Draw ECG paper-style grid (default ``True``).</td></tr>
    <tr><td><code>figsize</code></td><td><code>tuple</code></td><td>Figure size in inches (default ``(12, 3)``).</td></tr>
    <tr><td><code>ax</code></td><td><code>Axes | None</code></td><td>Existing axes to draw on. A new figure is created if ``None``.</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.plotting import plot_lead

result = plot_lead(...)
```

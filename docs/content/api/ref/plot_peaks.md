---
title: "plot_peaks"
weight: 34
---

`ecgdatakit.plotting.static.plot_peaks`

**Module:** Static Plots (matplotlib)

## Signature

```python
plot_peaks(lead, peaks=None, title=None, figsize=(12, 3), ax=None, *, fs=None)
```

## Description

Plot lead with R-peak markers and RR interval annotations.

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
from ecgdatakit.plotting import plot_peaks

result = plot_peaks(...)
```

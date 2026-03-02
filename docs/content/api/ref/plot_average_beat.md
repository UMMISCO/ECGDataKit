---
title: "plot_average_beat"
weight: 36
---

`ecgdatakit.plotting.static.plot_average_beat`

**Module:** Static Plots (matplotlib)

## Signature

```python
plot_average_beat(lead, peaks=None, before=0.2, after=0.4, figsize=(6, 4), ax=None, *, fs=None)
```

## Description

Plot ensemble-averaged beat with ±1 SD shading.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>Source ECG lead or raw signal array.</td></tr>
    <tr><td><code>peaks</code></td><td><code>NDArray | None</code></td><td>R-peak indices.</td></tr>
    <tr><td><code>before</code></td><td><code>float</code></td><td>Seconds before R-peak.</td></tr>
    <tr><td><code>after</code></td><td><code>float</code></td><td>Seconds after R-peak.</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.plotting import plot_average_beat

result = plot_average_beat(...)
```

---
title: "plot_beats"
weight: 35
---

`ecgdatakit.plotting.static.plot_beats`

**Module:** Static Plots (matplotlib)

## Signature

```python
plot_beats(lead, beats=None, peaks=None, overlay=True, figsize=(8, 5), ax=None, *, fs=None)
```

## Description

Plot segmented heartbeats.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>Source ECG lead or raw signal array.</td></tr>
    <tr><td><code>beats</code></td><td><code>list[Lead] | None</code></td><td>Pre-segmented beats. Segmented automatically if ``None``.</td></tr>
    <tr><td><code>peaks</code></td><td><code>NDArray | None</code></td><td>R-peak indices for segmentation.</td></tr>
    <tr><td><code>overlay</code></td><td><code>bool</code></td><td>``True``: overlay all beats; ``False``: waterfall display.</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.plotting import plot_beats

result = plot_beats(...)
```

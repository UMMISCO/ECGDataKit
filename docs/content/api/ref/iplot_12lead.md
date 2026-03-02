---
title: "iplot_12lead"
weight: 46
---

`ecgdatakit.plotting.interactive.iplot_12lead`

**Module:** Interactive Plots (plotly)

## Signature

```python
iplot_12lead(leads, record=None, duration=10.0, height=800)
```

## Description

Interactive 12-lead grid.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>leads</code></td><td><code>list[Lead] | ECGRecord</code></td><td>Leads (or record) to plot.</td></tr>
    <tr><td><code>record</code></td><td><code>ECGRecord | None</code></td><td>Optional record for header annotations.</td></tr>
    <tr><td><code>duration</code></td><td><code>float</code></td><td>Seconds per cell (default 10).</td></tr>
    <tr><td><code>height</code></td><td><code>int</code></td><td>Figure height in pixels.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.plotting import iplot_12lead

result = iplot_12lead(...)
```

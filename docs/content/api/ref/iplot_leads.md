---
title: "iplot_leads"
weight: 45
---

`ecgdatakit.plotting.interactive.iplot_leads`

**Module:** Interactive Plots (plotly)

## Signature

```python
iplot_leads(leads, peaks_dict=None, title=None, height=None)
```

## Description

Interactive stacked leads with synchronized X-axis zoom.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>leads</code></td><td><code>list[Lead] | ECGRecord</code></td><td>Leads to plot.</td></tr>
    <tr><td><code>peaks_dict</code></td><td><code>dict | None</code></td><td>``{label: peaks_array}`` for per-lead peak markers.</td></tr>
    <tr><td><code>title</code></td><td><code>str | None</code></td><td>Overall title.</td></tr>
    <tr><td><code>height</code></td><td><code>int | None</code></td><td>Figure height (auto-calculated if ``None``).</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.plotting import iplot_leads

result = iplot_leads(...)
```

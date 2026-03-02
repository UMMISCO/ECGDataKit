---
title: "plot_leads"
weight: 32
---

`ecgdatakit.plotting.static.plot_leads`

**Module:** Static Plots (matplotlib)

## Signature

```python
plot_leads(leads, peaks_dict=None, title=None, show_grid=True, figsize=(12, None), share_x=True)
```

## Description

Plot multiple leads stacked vertically.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>leads</code></td><td><code>list[Lead] | ECGRecord</code></td><td>Leads to plot.</td></tr>
    <tr><td><code>peaks_dict</code></td><td><code>dict | None</code></td><td>``{label: peaks_array}`` for per-lead R-peak markers.</td></tr>
    <tr><td><code>title</code></td><td><code>str | None</code></td><td>Overall figure title.</td></tr>
    <tr><td><code>show_grid</code></td><td><code>bool</code></td><td>Draw ECG paper-style grid.</td></tr>
    <tr><td><code>figsize</code></td><td><code>tuple</code></td><td>Width is fixed; height is auto-calculated (2 in per lead) when ``None``.</td></tr>
    <tr><td><code>share_x</code></td><td><code>bool</code></td><td>Share the x-axis across all subplots (default ``True``).</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.plotting import plot_leads

result = plot_leads(...)
```

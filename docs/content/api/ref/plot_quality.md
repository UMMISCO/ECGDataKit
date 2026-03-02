---
title: "plot_quality"
weight: 42
---

`ecgdatakit.plotting.static.plot_quality`

**Module:** Static Plots (matplotlib)

## Signature

```python
plot_quality(leads, figsize=(10, 5))
```

## Description

Signal quality dashboard: SQI bar chart per lead.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>leads</code></td><td><code>list[Lead] | ECGRecord</code></td><td>Leads to assess.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.plotting import plot_quality

result = plot_quality(...)
```

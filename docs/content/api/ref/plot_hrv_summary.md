---
title: "plot_hrv_summary"
weight: 41
---

`ecgdatakit.plotting.static.plot_hrv_summary`

**Module:** Static Plots (matplotlib)

## Signature

```python
plot_hrv_summary(rr_ms, figsize=(14, 8))
```

## Description

Combined HRV dashboard: tachogram, Poincaré, frequency bands, metrics.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>rr_ms</code></td><td><code>NDArray</code></td><td>RR intervals in milliseconds.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.plotting import plot_hrv_summary

result = plot_hrv_summary(...)
```

---
title: "plot_rr_tachogram"
weight: 39
---

`ecgdatakit.plotting.static.plot_rr_tachogram`

**Module:** Static Plots (matplotlib)

## Signature

```python
plot_rr_tachogram(rr_ms, figsize=(10, 3), ax=None)
```

## Description

Plot RR interval tachogram.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>rr_ms</code></td><td><code>NDArray</code></td><td>RR intervals in milliseconds.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.plotting import plot_rr_tachogram

result = plot_rr_tachogram(...)
```

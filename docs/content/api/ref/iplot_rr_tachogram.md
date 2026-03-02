---
title: "iplot_rr_tachogram"
weight: 49
---

`ecgdatakit.plotting.interactive.iplot_rr_tachogram`

**Module:** Interactive Plots (plotly)

## Signature

```python
iplot_rr_tachogram(rr_ms, height=300)
```

## Description

Interactive RR interval tachogram.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>rr_ms</code></td><td><code>NDArray</code></td><td>RR intervals in milliseconds.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.plotting import iplot_rr_tachogram

result = iplot_rr_tachogram(...)
```

---
title: "iplot_poincare"
weight: 50
---

`ecgdatakit.plotting.interactive.iplot_poincare`

**Module:** Interactive Plots (plotly)

## Signature

```python
iplot_poincare(rr_ms, height=500)
```

## Description

Interactive Poincaré plot with SD1/SD2 ellipse.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>rr_ms</code></td><td><code>NDArray</code></td><td>RR intervals in milliseconds.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.plotting import iplot_poincare

result = iplot_poincare(...)
```

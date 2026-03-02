---
title: "plot_poincare"
weight: 40
---

`ecgdatakit.plotting.static.plot_poincare`

**Module:** Static Plots (matplotlib)

## Signature

```python
plot_poincare(rr_ms, figsize=(6, 6), ax=None)
```

## Description

Poincaré plot: RR(n) vs RR(n+1) with SD1/SD2 ellipse.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>rr_ms</code></td><td><code>NDArray</code></td><td>RR intervals in milliseconds.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.plotting import plot_poincare

result = plot_poincare(...)
```

---
title: "poincare"
weight: 19
---

`ecgdatakit.processing.hrv.poincare`

**Module:** Heart Rate Variability

## Signature

```python
poincare(rr_ms)
```

## Description

Compute Poincaré plot descriptors (SD1, SD2).

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>rr_ms</code></td><td><code>NDArray</code></td><td>RR intervals in milliseconds.</td></tr>
  </tbody>
</table>

## Returns

<table>
  <thead><tr><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>dict</code></td><td>Keys: ``sd1``, ``sd2``, ``sd1_sd2_ratio``.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import poincare

result = poincare(...)
```

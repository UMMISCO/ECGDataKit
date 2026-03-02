---
title: "time_domain"
weight: 17
---

`ecgdatakit.processing.hrv.time_domain`

**Module:** Heart Rate Variability

## Signature

```python
time_domain(rr_ms)
```

## Description

Compute time-domain HRV metrics from RR intervals.

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
    <tr><td><code>dict</code></td><td>Keys: ``mean_rr``, ``sdnn``, ``rmssd``, ``sdsd``, ``nn50_count``, ``pnn50``, ``nn20_count``, ``pnn20``, ``hr_mean``, ``hr_std``.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import time_domain

result = time_domain(...)
```

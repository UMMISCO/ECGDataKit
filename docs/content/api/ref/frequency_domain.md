---
title: "frequency_domain"
weight: 18
---

`ecgdatakit.processing.hrv.frequency_domain`

**Module:** Heart Rate Variability

## Signature

```python
frequency_domain(rr_ms, method='welch', interp_fs=4.0)
```

## Description

Compute frequency-domain HRV metrics from RR intervals.

RR intervals are interpolated to a uniform time series at *interp_fs* Hz,
then the power spectral density is estimated.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>rr_ms</code></td><td><code>NDArray</code></td><td>RR intervals in milliseconds.</td></tr>
    <tr><td><code>method</code></td><td><code>str</code></td><td>PSD method (``&quot;welch&quot;``).</td></tr>
    <tr><td><code>interp_fs</code></td><td><code>float</code></td><td>Interpolation sampling rate in Hz (default 4.0).</td></tr>
  </tbody>
</table>

## Returns

<table>
  <thead><tr><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>dict</code></td><td>Keys: ``vlf_power``, ``lf_power``, ``hf_power``, ``lf_hf_ratio``, ``total_power``.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import frequency_domain

result = frequency_domain(...)
```

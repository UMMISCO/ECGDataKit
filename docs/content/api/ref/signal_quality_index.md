---
title: "signal_quality_index"
weight: 24
---

`ecgdatakit.processing.quality.signal_quality_index`

**Module:** Signal Quality

## Signature

```python
signal_quality_index(lead, *, fs=None)
```

## Description

Compute a composite signal quality index (SQI) in the range [0, 1].

Combines four sub-metrics: kurtosis SQI, power-ratio SQI,
R-peak regularity SQI, and baseline stability SQI.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>Input ECG lead or raw signal array.</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Returns

<table>
  <thead><tr><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>float</code></td><td>Score between 0.0 (unusable) and 1.0 (excellent).</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import signal_quality_index

result = signal_quality_index(...)
```

---
title: "normalize_zscore"
weight: 11
---

`ecgdatakit.processing.normalize.normalize_zscore`

**Module:** Normalization

## Signature

```python
normalize_zscore(lead, *, fs=None)
```

## Description

Normalize signal to zero mean and unit variance (z-score).

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>Input ECG lead or raw signal array.</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import normalize_zscore

result = normalize_zscore(...)
```

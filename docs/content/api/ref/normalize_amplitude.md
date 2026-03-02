---
title: "normalize_amplitude"
weight: 12
---

`ecgdatakit.processing.normalize.normalize_amplitude`

**Module:** Normalization

## Signature

```python
normalize_amplitude(lead, target_mv=1.0, *, fs=None)
```

## Description

Scale signal so that its maximum absolute amplitude equals *target_mv*.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>Input ECG lead or raw signal array.</td></tr>
    <tr><td><code>target_mv</code></td><td><code>float</code></td><td>Target peak amplitude (default 1.0).</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import normalize_amplitude

result = normalize_amplitude(...)
```

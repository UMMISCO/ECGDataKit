---
title: "classify_quality"
weight: 25
---

`ecgdatakit.processing.quality.classify_quality`

**Module:** Signal Quality

## Signature

```python
classify_quality(lead, *, fs=None)
```

## Description

Classify signal quality as a human-readable category.

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
    <tr><td><code>str</code></td><td>``&quot;excellent&quot;`` (SQI &gt; 0.8), ``&quot;acceptable&quot;`` (0.5--0.8), or ``&quot;unacceptable&quot;`` (&lt; 0.5).</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import classify_quality

result = classify_quality(...)
```

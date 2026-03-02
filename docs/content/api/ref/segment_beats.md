---
title: "segment_beats"
weight: 22
---

`ecgdatakit.processing.transforms.segment_beats`

**Module:** Transforms & Segmentation

## Signature

```python
segment_beats(lead, peaks=None, before=0.2, after=0.4, *, fs=None)
```

## Description

Segment individual heartbeats around R-peaks.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>Input ECG lead or raw signal array.</td></tr>
    <tr><td><code>peaks</code></td><td><code>NDArray | None</code></td><td>R-peak indices. Detected automatically if ``None``.</td></tr>
    <tr><td><code>before</code></td><td><code>float</code></td><td>Seconds before R-peak to include (default 0.2).</td></tr>
    <tr><td><code>after</code></td><td><code>float</code></td><td>Seconds after R-peak to include (default 0.4).</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Returns

<table>
  <thead><tr><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>list[Lead]</code></td><td>One Lead per beat, labelled ``&quot;{label}_beat_{i}&quot;``.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import segment_beats

result = segment_beats(...)
```

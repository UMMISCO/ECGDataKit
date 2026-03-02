---
title: "clean_ecg"
weight: 1
---

`ecgdatakit.processing.clean.clean_ecg`

**Module:** ECG Cleaning

## Signature

```python
clean_ecg(lead, method='default', *, fs=None, **kwargs)
```

## Description

Clean an ECG lead signal.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead</code></td><td><code>Lead | NDArray[np.float64]</code></td><td>Input ECG lead or raw signal array.</td></tr>
    <tr><td><code>method</code></td><td><code>str</code></td><td>Cleaning method: ``&quot;default&quot;``, ``&quot;biosppy&quot;``, ``&quot;neurokit2&quot;``, ``&quot;combined&quot;``, or ``&quot;deepfade&quot;``.</td></tr>
    <tr><td><code>fs</code></td><td><code>int | None</code></td><td>Sample rate in Hz.  Required when *lead* is a numpy array.</td></tr>
  </tbody>
</table>

## Returns

<table>
  <thead><tr><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>Lead</code></td><td>Cleaned lead (new object, original unchanged).</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import clean_ecg

result = clean_ecg(...)
```

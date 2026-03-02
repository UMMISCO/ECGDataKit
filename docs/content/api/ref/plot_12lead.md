---
title: "plot_12lead"
weight: 33
---

`ecgdatakit.plotting.static.plot_12lead`

**Module:** Static Plots (matplotlib)

## Signature

```python
plot_12lead(leads, record=None, paper_speed=25, amplitude=10, rhythm_lead='II', duration=10.0, figsize=(14, 10))
```

## Description

Standard 12-lead ECG grid layout.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>leads</code></td><td><code>list[Lead] | ECGRecord</code></td><td>Leads (or full record) to plot.</td></tr>
    <tr><td><code>record</code></td><td><code>ECGRecord | None</code></td><td>If provided, header with patient/device/measurement info is shown.</td></tr>
    <tr><td><code>paper_speed</code></td><td><code>float</code></td><td>Paper speed in mm/s (default 25).</td></tr>
    <tr><td><code>amplitude</code></td><td><code>float</code></td><td>Amplitude scale in mm/mV (default 10).</td></tr>
    <tr><td><code>rhythm_lead</code></td><td><code>str</code></td><td>Lead used for the full-length rhythm strip (default ``&quot;II&quot;``).</td></tr>
    <tr><td><code>duration</code></td><td><code>float</code></td><td>Seconds of signal to show per cell (default 10.0).</td></tr>
    <tr><td><code>figsize</code></td><td><code>tuple</code></td><td>Figure size.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.plotting import plot_12lead

result = plot_12lead(...)
```

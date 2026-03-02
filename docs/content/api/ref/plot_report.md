---
title: "plot_report"
weight: 43
---

`ecgdatakit.plotting.static.plot_report`

**Module:** Static Plots (matplotlib)

## Signature

```python
plot_report(record, figsize=(16, 20))
```

## Description

Comprehensive ECG report page.

Includes header with patient/device info, 12-lead grid,
rhythm strip, and quality indicators.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>record</code></td><td><code>ECGRecord</code></td><td>Full ECG record.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.plotting import plot_report

result = plot_report(...)
```

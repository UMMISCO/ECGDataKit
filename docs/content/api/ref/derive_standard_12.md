---
title: "derive_standard_12"
weight: 29
---

`ecgdatakit.processing.leads.derive_standard_12`

**Module:** Lead Derivation

## Signature

```python
derive_standard_12(lead_i, lead_ii, v1, v2, v3, v4, v5, v6)
```

## Description

Assemble a full 12-lead ECG, deriving III, aVR, aVL, aVF.

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead_i, lead_ii</code></td><td><code>Lead</code></td><td>Limb leads I and II.</td></tr>
    <tr><td><code>v1..v6</code></td><td><code>Lead</code></td><td>Precordial leads.</td></tr>
  </tbody>
</table>

## Returns

<table>
  <thead><tr><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>list[Lead]</code></td><td>12 leads in standard order: I, II, III, aVR, aVL, aVF, V1–V6.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import derive_standard_12

result = derive_standard_12(...)
```

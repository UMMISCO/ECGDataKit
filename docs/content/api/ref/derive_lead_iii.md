---
title: "derive_lead_iii"
weight: 27
---

`ecgdatakit.processing.leads.derive_lead_iii`

**Module:** Lead Derivation

## Signature

```python
derive_lead_iii(lead_i, lead_ii)
```

## Description

Derive Lead III from Leads I and II (Einthoven's law: III = II - I).

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>lead_i</code></td><td><code>Lead</code></td><td>Lead I signal.</td></tr>
    <tr><td><code>lead_ii</code></td><td><code>Lead</code></td><td>Lead II signal.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import derive_lead_iii

result = derive_lead_iii(...)
```

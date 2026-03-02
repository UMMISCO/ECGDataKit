---
title: "derive_augmented"
weight: 28
---

`ecgdatakit.processing.leads.derive_augmented`

**Module:** Lead Derivation

## Signature

```python
derive_augmented(lead_i, lead_ii)
```

## Description

Derive augmented limb leads aVR, aVL, aVF from Leads I and II.

## Returns

<table>
  <thead><tr><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>list[Lead]</code></td><td>[aVR, aVL, aVF] in that order.</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import derive_augmented

result = derive_augmented(...)
```
